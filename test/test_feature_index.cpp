#include "test.h"
#include "feature_index.h"
#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace DENSE_MULTICUT;

void test_exact_lookup(const size_t n, const size_t d, const std::string index_str)
{
    std::cout << "test exact lookup for " << n << " elements of dimension " << d << "\n";
    std::vector<float> features(n*d);
    std::mt19937 generator(0); // for deterministic behaviour
    std::uniform_real_distribution<float>  distr(-1.0, 1.0);

    for(size_t i=0; i<n*d; ++i)
        features[i] = distr(generator); 

    feature_index index = feature_index(d, n, features, index_str);

    {
        std::vector<faiss::Index::idx_t> all_indices(n); 
        std::iota(all_indices.begin(), all_indices.end(), 0);
        const auto [nns, distances] = index.get_nearest_nodes(all_indices);

        for(size_t i=0; i<n; ++i)
        {
            const auto [nn, dist] = index.get_nearest_node(i);
            test(nns[i] == nn);
            test(std::abs(distances[i] - dist) < 1e-7*d);
        }
    }

    // remove half the points and test again
    std::vector<faiss::Index::idx_t> remove_indices(n);
    std::iota(remove_indices.begin(), remove_indices.end(), 0);
    std::shuffle(remove_indices.begin(), remove_indices.end(), generator);
    for(size_t i=0; i<n/2; ++i)
        index.remove(remove_indices[i]);

    {
        std::vector<faiss::Index::idx_t> all_indices; 
        for(faiss::Index::idx_t idx=0; idx<n; ++idx)
            if(index.node_active(idx))
                all_indices.push_back(idx);

        const auto [nns, distances] = index.get_nearest_nodes(all_indices);

        for(size_t i=0; i<all_indices.size(); ++i)
        {
            const auto [nn, dist] = index.get_nearest_node(all_indices[i]);
            test(nns[i] == nn);
            test(std::abs(distances[i] - dist) < 1e-7*d);
            test(std::abs(dist - index.inner_product(i,nn)) < d*1e-7);
        }
    }
}

int main(int argc, char** argv)
{
    const std::vector<size_t> nr_nodes = {10,20,50,100,1000};
    const std::vector<size_t> nr_dims = {16,32,64,128,256,512,1024};
    for(const size_t n : nr_nodes)
        for(const size_t d : nr_dims)
            test_exact_lookup(n, d, "IDMap,Flat");
}
