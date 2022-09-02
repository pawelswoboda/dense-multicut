#include "feature_index.h"
#include <faiss/index_factory.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <unordered_map>
//#include <iostream>

namespace DENSE_MULTICUT {

    feature_index::feature_index(const size_t _d, const size_t n, const std::vector<float>& _features, const std::string& index_str)
        : d(_d),
        features(_features),
        index(index_factory(d, index_str.c_str(), faiss::MetricType::METRIC_INNER_PRODUCT)),
        nr_active(n)
    {
        //index = index_factory(d, index_str.c_str(), faiss::MetricType::METRIC_INNER_PRODUCT);
        std::vector<faiss::Index::idx_t> ids(n);
        std::iota(ids.begin(), ids.end(), 0);
        index->add_with_ids(n, features.data(), ids.data());

        active = std::vector<char>(n, true);
    }

    std::tuple<faiss::Index::idx_t, float> feature_index::get_nearest_node(const faiss::Index::idx_t id) const
    {
        if(can_remove)
        {
            float distance[2];
            faiss::Index::idx_t nns[2];
            index->search(1, features.data() + id*d, 2, distance, nns);
            assert(active[nns[0]]);
            assert(active[nns[1]]);
            if(nns[0] == id)
            {
                assert(nns[1] != id);
                return {nns[1], distance[1]};
            }
            else // duplicate entry
            {
                return {nns[0], distance[0]};
            }
        }
        else
        {
            for(size_t nr_lookups=2; nr_lookups<2*index->ntotal; nr_lookups*=2)
            {
                float distance[std::min(nr_lookups, size_t(index->ntotal))];
                faiss::Index::idx_t nns[std::min(nr_lookups, size_t(index->ntotal))];
                index->search(1, features.data() + id*d, std::min(nr_lookups, size_t(index->ntotal)), distance, nns);
                assert(std::is_sorted(distance, distance + std::min(nr_lookups, size_t(index->ntotal)), std::greater<float>()));
                for(size_t k=0; k<std::min(nr_lookups, size_t(index->ntotal)); ++k)
                    if(nns[k] < active.size() && nns[k] != id && active[nns[k]] == true)
                        return {nns[k], distance[k]};
            }
            throw std::runtime_error("Could not find nearest neighbor");
        }
    }

    std::tuple<std::vector<faiss::Index::idx_t>, std::vector<float>> feature_index::get_nearest_nodes(const std::vector<faiss::Index::idx_t>& nodes) const
    {
        assert(nodes.size() > 0);
        //std::cout << "[feature index] search nearest neighbors for " << nodes.size() << " nodes\n";

        if(can_remove)
        {
            std::vector<float> distances(2*nodes.size());
            std::vector<faiss::Index::idx_t> nns(2*nodes.size());
            std::vector<float> query_features(nodes.size()*d);
            for(size_t c=0; c<nodes.size(); ++c)
                for(size_t l=0; l<d; ++l)
                    query_features[c*d + l] = features[nodes[c]*d + l];

            index->search(nodes.size(), query_features.data(), 2, distances.data(), nns.data());

            std::vector<faiss::Index::idx_t> return_nns(nodes.size());
            std::vector<float> return_distances(nodes.size());
            for(size_t c=0; c<nodes.size(); ++c)
            {
                assert(active[nns[c*2]]);
                assert(active[nns[c*2+1]]);
                if(nns[2*c] == nodes[c])
                {
                    assert(nns[c*2+1] != nodes[c]);
                    std::tie(return_nns[c], return_distances[c]) = {nns[c*2+1], distances[c*2+1]};
                }
                else // duplicate entry
                {
                    std::tie(return_nns[c], return_distances[c]) = {nns[c*2], distances[c*2]};
                }
            }

            for(size_t i=0; i<return_nns.size(); ++i)
                assert(return_nns[i] != nodes[i]);
            return {return_nns, return_distances};
        }
        else
        {
            std::vector<faiss::Index::idx_t> return_nns(nodes.size());
            std::vector<float> return_distances(nodes.size());

            std::unordered_map<faiss::Index::idx_t, faiss::Index::idx_t> node_map;
            node_map.reserve(nodes.size());
            for(size_t c=0; c<nodes.size(); ++c)
                node_map.insert({nodes[c], c});

            for(size_t _nr_lookups=2; _nr_lookups<2+2*max_id_nr(); _nr_lookups*=2)
            {
                const size_t nr_lookups = std::min(_nr_lookups, size_t(index->ntotal));
                if(node_map.size() > 0)
                {
                    std::vector<faiss::Index::idx_t> cur_nodes;
                    for(const auto [node, idx] : node_map)
                        cur_nodes.push_back(node);
                    std::vector<faiss::Index::idx_t> nns(cur_nodes.size() * nr_lookups);
                    std::vector<float> distances(cur_nodes.size() * nr_lookups);

                    std::vector<float> query_features(node_map.size()*d);
                    for(size_t c=0; c<cur_nodes.size(); ++c)
                        for(size_t l=0; l<d; ++l)
                            query_features[c*d + l] = features[cur_nodes[c]*d + l];

                    index->search(cur_nodes.size(), query_features.data(), nr_lookups, distances.data(), nns.data());

                    for(size_t c=0; c<cur_nodes.size(); ++c)
                    {
                        for(size_t k=0; k<nr_lookups; ++k)
                        {
                            if(nns[c*nr_lookups + k] < active.size() && nns[c*nr_lookups + k] != cur_nodes[c] && active[nns[nr_lookups*c + k]] == true)
                            {
                                assert(node_map.count(cur_nodes[c]) > 0);
                                return_nns[node_map[cur_nodes[c]]] = nns[c*nr_lookups + k];
                                return_distances[node_map[cur_nodes[c]]] = distances[c*nr_lookups + k];
                                node_map.erase(cur_nodes[c]);
                                break;
                            }
                        }
                    }

                }
            }

            for(size_t i=0; i<return_nns.size(); ++i)
                assert(return_nns[i] != nodes[i]);
            return {return_nns, return_distances};
        }
    }

    void feature_index::remove(const faiss::Index::idx_t i)
    {
        assert(i < active.size());
        assert(active[i] == true);
        active[i] = false;
        nr_active--;

        if(can_remove)
        {
            faiss::IDSelectorRange to_remove(i,i+1);
            index->remove_ids(to_remove);
        }
    }

    faiss::Index::idx_t feature_index::merge(const faiss::Index::idx_t i, const faiss::Index::idx_t j)
    {
        assert(i != j);
        assert(i < active.size());
        assert(j < active.size());

        active[i] = false;
        active[j] = false;

        nr_active--;

        if(can_remove)
        {
            faiss::IDSelectorRange to_remove(i,i+1);
            index->remove_ids(to_remove);
            to_remove = faiss::IDSelectorRange(j,j+1);
            index->remove_ids(to_remove);
        }

        const faiss::Index::idx_t new_id = features.size()/d;
        for(size_t l=0; l<d; ++l)
            features.push_back(features[i*d + l] + features[j*d + l]);
        index->add_with_ids(1, features.data() + new_id*d, &new_id);
        assert(active.size() == new_id);
        active.push_back(true);
        return new_id;
    }

    float feature_index::inner_product(const faiss::Index::idx_t i, const faiss::Index::idx_t j)
    {
        assert(i < active.size());
        assert(j < active.size());
        float x = 0.0;
        for(size_t l=0; l<d; ++l)
            x += features[i*d+l]*features[j*d+l];
        return x;
    }

    bool feature_index::node_active(const faiss::Index::idx_t idx) const
    {
        assert(idx < active.size());
        return active[idx] == true;
    }

    size_t feature_index::max_id_nr() const 
    { 
        assert(active.size() > 0);
        return active.size()-1;
    }

    size_t feature_index::nr_nodes() const
    {
        assert(nr_active == std::count(active.begin(), active.end(), true));
        return nr_active;
    }

}
