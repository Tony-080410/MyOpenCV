#pragma once

#include <mycv/core/mat.hpp>
#include <string>

namespace mycv {

Mat imread(const std::string &path); // P5->U8C1,P6->U8C3
void imwrite(const std::string &path, const Mat &m); // U8C1->P5,U8C3->P6

} // namespace mycv