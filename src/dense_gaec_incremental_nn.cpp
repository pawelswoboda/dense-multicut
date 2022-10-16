#include "dense_gaec_incremental_nn.h"
#include "feature_index.h"
#include "incremental_nns.h"
#include "dense_multicut_utils.h"
#include "union_find.hxx"
#include "time_measure_util.h"

#include <vector>
#include <queue>
#include <numeric>
#include <random>
#include <iostream>

#include <faiss/index_factory.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetaIndexes.h>

#include <faiss/IndexHNSW.h>

#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <faiss/impl/AuxIndexStructures.h>

namespace DENSE_MULTICUT {

    std::vector<size_t> dense_gaec_incremental_nn(const size_t n, const size_t d, std::vector<float> features, const size_t k)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        feature_index index(d, n, features, "Flat");
        assert(features.size() == n*d);

        std::cout << "[dense gaec incremental nn] Find multicut for " << n << " nodes with features of dimension " << d << "\n";

        double multicut_cost = cost_disconnected(n, d, features);

        const size_t max_nr_ids = 2*n;
        union_find uf(max_nr_ids);

        incremental_nns nn_graph;
        using pq_type = std::tuple<float, std::array<faiss::Index::idx_t,2>>;
        auto pq_comp = [](const pq_type& a, const pq_type& b) { return std::get<0>(a) < std::get<0>(b); };
        std::priority_queue<pq_type, std::vector<pq_type>, decltype(pq_comp)> pq(pq_comp);
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("Initial KNN construction");
            std::vector<faiss::Index::idx_t> all_indices(n);
            std::iota(all_indices.begin(), all_indices.end(), 0);
            const auto [nns, distances] = index.get_nearest_nodes(all_indices, k);
            std::cout<<"[dense gaec incremental nn] Initial NN search complete\n";
            nn_graph = incremental_nns(nns, distances, n, k);
            size_t index_1d = 0;
            for(size_t i=0; i<n; ++i)
            {
                for(size_t i_k=0; i_k < k; ++i_k, ++index_1d)
                {
                    if(distances[index_1d] > 0.0)
                    {
                        pq.push({distances[index_1d], {i, nns[index_1d]}});
                    }
                }
            }
        }

        // iteratively find pairs of features with highest inner product
        while(!pq.empty()) {
            const auto [distance, ij] = pq.top();
            pq.pop();
            assert(distance > 0.0);
            const auto [i,j] = ij;
            assert(i != j);
            // check if edge is still present in contracted graph. This is true if both endpoints have not been contracted
            if(index.node_active(i) && index.node_active(j))
            {
                //std::cout << "[dense multicut " << index_str << "] contracting edge " << i << " and " << j << " with edge cost " << distance << "\n";
                // contract edge:
                const size_t new_id = index.merge(i,j);

                uf.merge(i, new_id);
                uf.merge(j, new_id);
                const std::map<size_t, float> nn_ij = nn_graph.merge_nodes(i, j, new_id, index);
                multicut_cost -= distance;

                // find new nearest neighbor
                if(index.nr_nodes() > 1)
                    for (auto const& [nn_new, new_cost] : nn_ij)
                        pq.push({new_cost, {new_id, nn_new}});
            }
        }

        std::cout << "[dense gaec incremental nn] final nr clusters = " << uf.count() - (max_nr_ids - index.max_id_nr()-1) << "\n";
        std::cout << "[dense gaec incremental nn] final multicut cost = " << multicut_cost << "\n";

        std::vector<size_t> component_labeling(n);
        for(size_t i=0; i<n; ++i)
            component_labeling[i] = uf.find(i);
        return component_labeling;
    }
}
