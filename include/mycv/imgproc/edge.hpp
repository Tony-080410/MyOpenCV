#pragma once

#include <mycv/core/mat.hpp>

namespace mycv {

// Canny 边缘检测。（src 必须 U8C1。。。
// L2gradient=true 时幅值用 gx²+gy²（省开方），阈值也跟着平方
Mat canny(const Mat &src, double low, double high, bool L2gradient = false);

} // namespace mycv
