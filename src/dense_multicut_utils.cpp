#include "dense_multicut_utils.h"
#include <iostream>

namespace DENSE_MULTICUT {

    double cost_disconnected(const size_t n, const size_t d, const std::vector<float>& features)
    {
        std::vector<double> feature_sum(d);
        for(size_t i=0; i<n; ++i)
            for(size_t l=0; l<d; ++l)
                feature_sum[l] += features[i*d+l];

        double cost = 0.0;

        for(size_t l=0; l<d; ++l)
            cost += feature_sum[l] * feature_sum[l];

        // remove diagonal entries (self-edge)
        for(size_t i=0; i<n; ++i)
            for(size_t l=0; l<d; ++l)
                cost -= features[i*d+l]*features[i*d+l];

        std::cout << "disconnected multicut cost = " << cost/2.0 << "\n";
        return cost/2.0;
    }

}
