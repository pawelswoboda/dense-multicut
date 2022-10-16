#include "incremental_nns.h"
#include "time_measure_util.h"
#include <limits>
#include <cstddef>

namespace DENSE_MULTICUT {

    incremental_nns::incremental_nns(const std::vector<faiss::Index::idx_t>& nns, const std::vector<float>& nns_distances, const size_t n, const size_t k)
    {
        // Store as undirected graph.
        nn_graph_ = std::vector<std::map<size_t, float>>(2 * n);
        k_ = k;

        size_t index_1d = 0;
        for (size_t i = 0; i != n; ++i)
        {
            for (size_t i_n = 0; i_n != k; ++i_n, ++index_1d)
            {
                const float current_distance = nns_distances[index_1d];
                if (current_distance < 0)
                    continue;

                const size_t j = nns[index_1d];
                nn_graph_[i].try_emplace(j, current_distance);
                nn_graph_[j].try_emplace(i, current_distance);
            }
        }
    }

    std::map<size_t, float> incremental_nns::merge_nodes(const size_t i, const size_t j, const size_t new_id, const feature_index& index)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        const size_t root = nn_graph_[i].size() >= nn_graph_[j].size() ? i: j;
        const size_t other = root == i ? j : i;
        
        std::map<size_t, float> nn_ij;
        
        // iterate over kNNs of root:
        for (auto const& [nn_root, cost_root] : nn_graph_[root])
        {
            if (nn_root == other)
                continue;
            // check if nn_root is also in kNN(other):
            const auto nn_other_iter = nn_graph_[other].find(nn_root);
            if (nn_other_iter != nn_graph_[other].end())
                nn_ij.try_emplace(nn_root, cost_root + nn_other_iter->second);
            else
            {
                // Compute cost between other and nn_root and add.
                const float new_cost = cost_root + index.inner_product(nn_root, other); 
                if (new_cost > 0.0)
                    nn_ij.try_emplace(nn_root, new_cost);
            }
            // Remove root as neighbour of nn_root:
            nn_graph_[nn_root].erase(root);
        }

        // Now iterate over kNNs of other:
        for (auto const& [nn_other, cost_other] : nn_graph_[other])
        {
            if (nn_other == root)
                continue;
            // Skip if nn_other is also in kNN(root) as already considered above.
            if (nn_graph_[root].find(nn_other) == nn_graph_[root].end())
            {
                const float new_cost = cost_other + index.inner_product(nn_other, root); // Compute cost between root and nn_other and add.
                if (new_cost > 0.0)
                    nn_ij.try_emplace(nn_other, new_cost);
            }
            // Remove other as neighbour of nn_other:
            nn_graph_[nn_other].erase(other);
        }

        // If no new neighbours are found within KNNs of i and j, then search in whole graph for k_ many nearest neighbours.
        if (nn_ij.size() == 0)
        {
            std::cout<<"[incremental nns] Performing brute force NN search on "<<index.nr_nodes()<<" nodes \n";
            const std::vector<faiss::Index::idx_t> new_id_to_search = {new_id};
            const auto [nns, distances] = index.get_nearest_nodes(new_id_to_search, std::min(k_, index.nr_nodes() - 1));
            for (int idx = 0; idx != nns.size(); ++idx)
            {
                const float current_distance = distances[idx];
                if (current_distance > 0.0)
                    nn_ij.try_emplace(nns[idx], current_distance);
            }
        }

        // TODO: Remove root and other nodes? Perhaps not necessary since root and other node become 'inactive' anyway.
        // nn_graph_[root] = std::map<size_t, float>();
        // nn_graph_[other] = std::map<size_t, float>();

        // Create new node with id 'new_id' and add its neighbours:
        nn_graph_[new_id] = nn_ij;
        // Also add bidirectional edges:
        for (auto const& [nn_new, new_cost] : nn_ij)
            nn_graph_[nn_new].try_emplace(new_id, new_cost);
        
        return nn_ij;
    }
}