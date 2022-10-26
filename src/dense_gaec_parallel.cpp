#include "feature_index.h"
#include "dense_gaec_parallel.h"
#include "maximum_matching_greedy.h"
#include "dense_multicut_utils.h"
#include "time_measure_util.h"
#include "union_find.hxx"
#include <string>
#include <queue>

namespace DENSE_MULTICUT {

    std::vector<size_t> dense_gaec_parallel_impl(const size_t n, const size_t d, std::vector<float> features, const std::string index_str, const bool track_dist_offset)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        feature_index index(d, n, features, index_str, track_dist_offset);
        assert(features.size() == n*d);

        std::cout << "[dense gaec parallel " << index_str << "] Find multicut for " << n << " nodes with features of dimension " << d << "\n";

        double multicut_cost = cost_disconnected(n, d, features);

        const size_t max_nr_ids = 2*n;
        union_find uf(max_nr_ids);

        using pq_type = std::tuple<float, std::array<faiss::Index::idx_t,2>>;
        auto pq_comp = [](const pq_type& a, const pq_type& b) { return std::get<0>(a) < std::get<0>(b); };
        std::priority_queue<pq_type, std::vector<pq_type>, decltype(pq_comp)> pq(pq_comp);

        size_t iter = 0;
        for(; index.nr_nodes() > 0; ++iter)
        {
            std::vector<faiss::Index::idx_t> all_active_indices;
            for(faiss::Index::idx_t idx=0; idx<=index.max_id_nr(); ++idx)
                if(index.node_active(idx))
                    all_active_indices.push_back(idx);

            // TODO: this should be already registered by index.nr_nodes() above
            if(all_active_indices.size() == 0)
                break;

            const auto [nns, distances] = index.get_nearest_nodes(all_active_indices);
            if(*std::max_element(distances.begin(), distances.end()) <= 0.0)
                break;

            // find maximal matching of selected nearest neighbors
            std::vector<faiss::Index::idx_t> i;
            std::vector<faiss::Index::idx_t> j;
            std::vector<float> positive_distances;
            for(size_t c=0; c<all_active_indices.size(); ++c)
            {
                if(distances[c] > 0.0)
                {
                    i.push_back(all_active_indices[c]);
                    j.push_back(nns[c]);
                    positive_distances.push_back(distances[c]);
                }
            }

            const std::vector<std::array<size_t,2>> matching = maximum_matching_greedy(i.begin(), i.end(), j.begin(), positive_distances.begin());

            //std::cout << "[dense gaec parallel " << index_str << "] matching gave " << matching.size() << " edges to contract\n";

            for(const auto [i,j] : matching)
            {
                //std::cout << "[dense gaec parallel] contract edge " << i << " <-> " << j << " with edge cost " << index.inner_product(i,j) << "\n";
                const faiss::Index::idx_t new_id = index.merge(i,j);
                multicut_cost -= index.inner_product(i,j);
                uf.merge(i, new_id);
                uf.merge(j, new_id);
            }
        }

        const size_t nr_contracted_edges = n - (uf.count() - (max_nr_ids - index.max_id_nr()-1)); 
        std::cout << "[dense gaec parallel " << index_str << "] "
            << "final nr clusters = " << n - nr_contracted_edges 
            << " after " << iter << " iterations, i.e. " << nr_contracted_edges/double(iter) << " contractions per iteration\n";
        std::cout << "[dense gaec parallel " << index_str << "] final multicut cost = " << multicut_cost << "\n";

        std::vector<size_t> component_labeling(n);
        for(size_t i=0; i<n; ++i)
            component_labeling[i] = uf.find(i);
        return component_labeling;
    }

    std::vector<size_t> dense_gaec_parallel_flat_index(const size_t n, const size_t d, std::vector<float> features, const bool track_dist_offset)
    {
        std::cout << "Dense parallel GAEC with flat index\n";
        return dense_gaec_parallel_impl(n, d, features, "Flat", track_dist_offset);
    }

    std::vector<size_t> dense_gaec_parallel_hnsw(const size_t n, const size_t d, std::vector<float> features, const bool track_dist_offset)
    {
        std::cout << "Dense parallel GAEC with HNSW index\n";
        return dense_gaec_parallel_impl(n, d, features, "HNSW", track_dist_offset);
    }
}
