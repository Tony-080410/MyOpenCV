#include "mycv/core/mat.hpp"
#include <mycv/imgproc/color.hpp>

#include <cstdint>
#include <stdexcept>

#include <mycv/core/check.hpp>
#include <mycv/core/types.hpp>

namespace mycv {
namespace {

// 以下抄OpenCV。。。
// 15 位定点系数，和 OpenCV 4.10 color.simd_helpers.hpp 一致。
// 三者之和必须等于 2^15，否则纯白会算成 256 溢出

constexpr int BY15 = 3735; // 0.114
constexpr int GY15 = 19235; // 0.587
constexpr int RY15 = 9798; // 0.299
constexpr int GRAY_SHIFT = 15;
constexpr int GRAY_ROUND = 1 << (GRAY_SHIFT - 1); // 16384，加半个单位再右移

} // namespace

Mat cvtColor(const Mat &src, int code) {
    if (code != COLOR_BGR2GRAY)
        throw std::runtime_error("cvtColor: 未实现的 code " + std::to_string(code));

    MYCV_ASSERT(!src.empty());
    MYCV_ASSERT(src.type == U8C3); // 目前只吃 8 位三通道

    Mat dst;
    dst.create(src.rows, src.cols, U8C1);

    for (int r = 0; r < src.rows; ++r) {
        const uint8_t *s = src.ptr(r); // s 指向这一行第一个像素的 B
        uint8_t *d = dst.ptr(r);
        for (int c = 0; c < src.cols; ++c, s += 3) {
            d[c] = static_cast<uint8_t>((s[0] * BY15 + s[1] * GY15 + s[2] * RY15 + GRAY_ROUND) >>
                                        GRAY_SHIFT);
        }
    }
    return dst;
}

} // namespace mycv
