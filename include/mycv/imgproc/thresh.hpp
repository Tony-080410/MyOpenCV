#pragma once

#include <mycv/core/mat.hpp>
namespace mycv {

enum ThreshType {
    THRESH_BINARY = 0,
    THRESH_BINARY_INV = 1,
    THRESH_TRUNC = 2,
    THRESH_TOZERO = 3,
    THRESH_TOZERO_INV = 4,
};

// 目前支持 U8C1/U8C3,多通道单独处理
Mat threshold(const Mat &src, double thresh, double maxval, int type);

} // namespace mycv