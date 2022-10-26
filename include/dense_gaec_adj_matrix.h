#include <vector>
#include <cstddef>

namespace DENSE_MULTICUT {

    std::vector<size_t> dense_gaec_adj_matrix(const size_t n, const size_t d, std::vector<float> features, const bool track_dist_offset = false);

}

