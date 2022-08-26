#include "dense_gaec.h"
#include "dense_multicut_utils.h"
#include "union_find.hxx"

#include <vector>
#include <queue>
#include <numeric>
#include <iostream>

#include <faiss/IndexFlat.h>
#include <faiss/MetaIndexes.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <faiss/impl/AuxIndexStructures.h>

namespace DENSE_MULTICUT {

    std::vector<size_t> dense_gaec(const size_t n, const size_t d, std::vector<float> features)
    {
        double multicut_cost = cost_disconnected(n, d, features);

        // build up database of all features
        faiss::IndexFlatIP underlying_index(d);
        faiss::IndexIDMapTemplate<faiss::IndexFlatIP> index(&underlying_index);
        std::vector<faiss::Index::idx_t> ids(n, faiss::MetricType::METRIC_INNER_PRODUCT);
        std::iota(ids.begin(), ids.end(), 0);
        index.add_with_ids(n, features.data(), ids.data());
        std::cout << "[dense gaec] Added " << index.ntotal << " indices to table\n";

        auto get_nearest_node = [&](const faiss::Index::idx_t id) -> std::tuple<faiss::Index::idx_t, float> {
            float distance[2];
            faiss::Index::idx_t nns[2];
            index.search(1, features.data() + id*d, 2, distance, nns);
            if(nns[0] == id)
            {
                assert(nns[1] != id);
                return {nns[1], distance[1]};
            }
            else // duplicate entry
            {
                return {nns[0], distance[0]};
            }
        };

        using pq_type = std::tuple<float, std::array<faiss::Index::idx_t,2>>;
        auto pq_comp = [](const pq_type& a, const pq_type& b) { return std::get<0>(a) < std::get<0>(b); };
        std::priority_queue<pq_type, std::vector<pq_type>, decltype(pq_comp)> pq(pq_comp);

        const size_t max_nr_ids = 2*n;
        std::vector<std::vector<u_int32_t>> pq_pair(max_nr_ids);

        // TODO: search all entries simultaneously
        for(faiss::Index::idx_t i=0; i<n; ++i)
        {
            const auto [nns, distance] = get_nearest_node(i);
            if(distance > 0.0)
            {
                pq.push({distance, {i,nns}});
                pq_pair[nns].push_back(i);
                //std::cout << "[dense gaec] push initial shortest edge " << i << " <-> " << nns << " with cost " << distance << "\n";
            }
        }
        //std::cout << "[dense gaec] Added " << pq.size() << " initial elements to priority queue\n";

        std::vector<char> active(n, 1);
        active.reserve(max_nr_ids);
        union_find uf(max_nr_ids);

        // iteratively find pairs of features with highest inner product
        while(!pq.empty()) {
            const auto [distance, ij] = pq.top();
            pq.pop();
            assert(distance > 0.0);
            const auto [i,j] = ij;
            assert(i != j);
            // check if edge is still present in contracted graph. This is true if both endpoints have not been contracted
            assert(i < active.size() && j < active.size());
            if(active[i] && active[j])
            { 
                std::cout << "[dense multicut] contracting edge " << i << " and " << j << " with edge cost " << distance << "\n";
                // contract edge:
                // Remove original two features from index.
                faiss::IDSelectorRange to_remove(i,i+1);
                index.remove_ids(to_remove);
                active[i] = false;
                to_remove = faiss::IDSelectorRange(j,j+1);
                index.remove_ids(to_remove);
                active[j] = false;

                // add new feature corresponding to contracted component
                const faiss::Index::idx_t new_id = features.size()/d;
                assert(new_id < max_nr_ids);
                for(size_t l=0; l<d; ++l)
                    features.push_back(features[i*d + l] + features[j*d + l]);
                index.add_with_ids(1, features.data() + new_id*d, &new_id);
                assert(active.size() == new_id);
                active.push_back(true);

                uf.merge(i, new_id);
                uf.merge(j, new_id);

                multicut_cost -= distance;

                // find new nearest neighbor
                const auto [nns, distance] = get_nearest_node(new_id);
                if(distance > 0.0)
                {
                    pq.push({distance, {new_id,nns}});
                    pq_pair[nns].push_back(new_id);
                }

                // check nearest neighbors of i and j and see whether they need new nearest neighbors
                {
                    auto tmp_nodes = pq_pair[i];
                    pq_pair[i].clear();
                    for(const size_t k : tmp_nodes)
                    {
                        if(active[k])
                        {
                            const auto [nns, distance] = get_nearest_node(k);
                            if(distance > 0.0)
                            {
                                pq.push({distance, {k, nns}});
                                pq_pair[nns].push_back(k);
                            }
                        }
                    }
                }
                {
                    auto tmp_nodes = pq_pair[i];
                    pq_pair[i].clear();
                    for(const size_t k : pq_pair[j])
                    {
                        if(active[k])
                        {
                            const auto [nns, distance] = get_nearest_node(k);
                            if(distance > 0.0)
                            {
                                pq.push({distance, {k, nns}});
                                pq_pair[nns].push_back(k);
                            }
                        }
                    }
                }
            }
        }

        std::cout << "[dense gaec] final nr clusters = " << uf.count() - (max_nr_ids - active.size()) << "\n";
        std::cout << "[dense gaec] final multicut cost = " << multicut_cost << "\n";

        std::vector<size_t> component_labeling(n);
        for(size_t i=0; i<n; ++i)
            component_labeling[i] = uf.find(i);
        return component_labeling;
    }

}
