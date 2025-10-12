#ifndef FERM_HPP
#define FERM_HPP

#include <ferm/ferm_export.hpp>

namespace ferm {

FERM_EXPORT float
  europeanCall(float *spot, float *strike, float *rate, float *volatility, float *maturity, int *num_paths);

}// namespace ferm

#endif
