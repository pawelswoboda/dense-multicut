#include <faiss/Index.h>
#include "feature_index.h"
#include <vector>
#include <tuple>
#include <memory>
#include <map>

namespace DENSE_MULTICUT {

    class incremental_nns {
        public:
            incremental_nns() {}
            incremental_nns(const std::vector<faiss::Index::idx_t>& nns, const std::vector<float>& nns_distances, const size_t n, const size_t k);

            // Merges i, j to a single node with new_id and return neighbours of this single node and their associated edge costs.
            std::map<size_t, float> merge_nodes(const size_t i, const size_t j, const size_t new_id, const feature_index& index);

        private:
            std::vector<std::map<size_t, float>> nn_graph_;
            size_t k_;
    };
}