#include <cmath>
#include <cstdint>
#include <mycv/core/check.hpp>
#include <mycv/core/types.hpp>
#include <mycv/imgproc/filter.hpp>

namespace mycv {
namespace {

// BORDER_REFLECT_101: gfedcb|abcdefgh|gfedcba
int reflect101(int i, int n) {
    while (i < 0 || i >= n) {
        if (i < 0)
            i = -i;
        else
            i = 2 * n - i - 2;
    }
    return i;
}

} // namespace

Mat convolve2D(const Mat &src, const Mat &kernel, Point anchor) {
    MYCV_ASSERT(!src.empty());
    MYCV_ASSERT(!kernel.empty());
    MYCV_ASSERT(depthOf(src.type) == DEPTH_8U);
    MYCV_ASSERT(depthOf(kernel.type) == DEPTH_32F);
    MYCV_ASSERT(channels(kernel.type) == 1);

    // 只有 -1 表示"取核中心"，其他负数要落到下面的断言里
    // (2026.11.20 15:48 记录）原来写的是 < 0，结果 anchor={-5,0} 也被当成默认值，断言永远不触发
    const int ax = anchor.x == -1 ? kernel.cols / 2 : anchor.x;
    const int ay = anchor.y == -1 ? kernel.rows / 2 : anchor.y;
    MYCV_ASSERT(ax >= 0 && ax < kernel.cols);
    MYCV_ASSERT(ay >= 0 && ay < kernel.rows);

    const int cn = channels(src.type);

    Mat dst;
    dst.create(src.rows, src.cols, src.type);

    for (int y = 0; y < src.rows; ++y) {
        uint8_t *d = dst.ptr(y);
        for (int x = 0; x < src.cols; ++x) {
            for (int c = 0; c < cn; ++c) {
                double sum = 0.0;
                for (int ky = 0; ky < kernel.rows; ++ky) {
                    const uint8_t *row = src.ptr(reflect101(y + ky - ay, src.rows));
                    const float *krow = kernel.ptr<float>(ky);
                    for (int kx = 0; kx < kernel.cols; ++kx) {
                        sum += static_cast<double>(krow[kx]) *
                               row[reflect101(x + kx - ax, src.cols) * cn + c];
                    }
                }
                long v = std::lrint(sum); // 银行家舍入，和 cvRound 一致
                if (v < 0)
                    v = 0;
                else if (v > 255)
                    v = 255;
                d[x * cn + c] = static_cast<uint8_t>(v);
            }
        }
    }
    return dst;
}

Mat blur(const Mat &src, Size ksize) {
    MYCV_ASSERT(!src.empty());
    if (ksize.width <= 0 || ksize.height <= 0 || ksize.width % 2 == 0 || ksize.height % 2 == 0)
        throw std::runtime_error("blur: ksize 必须是正的奇数");

    Mat k;
    k.create(ksize.height, ksize.width, F32C1);
    const float w = 1.0f / static_cast<float>(ksize.width * ksize.height);
    for (int r = 0; r < k.rows; ++r)
        for (int c = 0; c < k.cols; ++c)
            k.at<float>(r, c) = w;

    return convolve2D(src, k);
}

Mat sobel(const Mat &src, int dx, int dy, int ksize) {
    MYCV_ASSERT(!src.empty());
    if (ksize != 3) throw std::runtime_error("sobel: 目前只支持 ksize=3");
    if (dx < 0 || dy < 0 || dx + dy != 1) throw std::runtime_error("sobel: 只支持 dx+dy==1");

    // OpenCV 的 Sobel 不做归一化，核和是 0；8U 输出会大量饱和到 0/255
    static const float kx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    static const float ky[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    const float *src9 = (dx == 1) ? kx : ky;

    Mat k;
    k.create(3, 3, F32C1);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            k.at<float>(r, c) = src9[r * 3 + c];

    return convolve2D(src, k);
}

} // namespace mycv
