#include <cassert>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include "time_measure_util.h"

namespace DENSE_MULTICUT {

    template<typename I_ITERATOR, typename J_ITERATOR, typename COST_ITERATOR>
    std::vector<std::array<size_t,2>> maximum_matching_greedy(I_ITERATOR i_begin, I_ITERATOR i_end, J_ITERATOR j_begin, COST_ITERATOR cost_begin)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        const size_t m = std::distance(i_begin, i_end);
        const size_t n = std::max( *std::max_element(i_begin, i_end), *std::max_element(j_begin, j_begin+m)) + 1;

        //std::cout << "[maximum matching greedy] input:";
        //for(size_t e=0; e<m; ++e)
        //    std::cout << " (" << *(i_begin+e) << "," << *(j_begin+e) << ";" << *(cost_begin+e) << ")";
        //std::cout << "\n";

        std::vector<std::array<size_t,2>> matching;

        std::vector<size_t> order(m);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](const size_t i, const size_t j) { return *(cost_begin+i) > *(cost_begin+j); });

        //std::cout << "[maximum matching greedy] order:";
        //for(size_t e=0; e<order.size(); ++e)
        //    std::cout << " " << order[e];
        //std::cout << "\n";

        for(size_t e=1; e<m; ++e)
            assert(*(cost_begin+order[e-1]) >= *(cost_begin+order[e]));

        std::vector<char> taken(n, false);
        for(size_t e=0; e<m; ++e)
        {
            const size_t i = *(i_begin+order[e]);
            const size_t j = *(j_begin+order[e]);
            assert(i != j);
            assert(i<taken.size());
            assert(j<taken.size());
            if(!taken[i] && !taken[j])
            {
                taken[i] = true;
                taken[j] = true;
                matching.push_back({i,j});
            }
        }

        return matching;
    }

}
