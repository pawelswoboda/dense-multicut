#include <vector>
#include <cstddef>

namespace DENSE_MULTICUT {

    double cost_disconnected(const size_t n, const size_t d, const std::vector<float>& features, const float dist_offset = 0.0f);
    std::vector<float> append_dist_offset_in_features(const std::vector<float>& features, const float dist_offset, const size_t n, const size_t d);

}
