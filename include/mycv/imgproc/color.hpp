#pragma once

#include <mycv/core/mat.hpp>

namespace mycv {
// code 值照抄 OpenCV，以后写适配层能直接传
enum ColorCode { COLOR_BGR2GRAY = 6 };

// 只支持 U8C3 -> U8C1。其他 code / 类型抛异常
Mat cvtColor(const Mat &src, int code);

} // namespace mycv
