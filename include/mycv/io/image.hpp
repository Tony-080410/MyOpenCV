#pragma once

#include <string>

#include <mycv/core/mat.hpp>

namespace mycv {
// 按魔数自动识别：P5/P6 -> PGM/PPM，FFD8 -> JPEG。失败抛 runtime_error
// P5 -> U8C1，P6/JPEG 彩色 -> U8C3（通道顺序统一成 BGR）
Mat imread(const std::string& path);

// 按扩展名选格式：.jpg/.jpeg -> JPEG，其余 -> PGM/PPM。失败抛 runtime_error
// jpegQuality 只在 JPEG 时有意义，范围 1..100
void imwrite(const std::string& path, const Mat& m, int jpegQuality = 95);
} // namespace mycv
