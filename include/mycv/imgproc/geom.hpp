#pragma once

#include <mycv/core/mat.hpp>
#include <mycv/core/types.hpp>
namespace mycv {

enum InterpFlag { INTER_NEAREST = 0, INTER_LINEAR = 1 };
Mat resize(const Mat &src, Size dstSize, int interpolation = INTER_LINEAR);

} // namespace mycv