#include <vector>
#include <cstddef>
#include <string>
namespace DENSE_MULTICUT {

    std::vector<size_t> dense_gaec_incremental_nn(const size_t n, const size_t d, std::vector<float> features, const size_t k, const std::string index_type = "Flat", const bool track_dist_offset = false);
}
