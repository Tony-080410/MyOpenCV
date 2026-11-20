#include <mycv/core/check.hpp>
#include <mycv/core/mat.hpp>
#include <mycv/imgproc/morph.hpp>
namespace mycv {
Mat getStructuringElement(int shape, Size ksize) {
    MYCV_ASSERT(ksize.width > 0 && ksize.height > 0);
    if (shape != MORPH_RECT && shape != MORPH_CROSS)
        throw std::runtime_error("getStructuringElement: 目前只支持 RECT 和 CROSS");

    Mat k;
    k.create(ksize.height, ksize.width, U8C1);

    const int cx = ksize.width / 2;
    const int cy = ksize.height / 2;

    for (int r = 0; r < k.rows; ++r) {
        uint8_t *p = k.ptr(r);
        for (int c = 0; c < k.cols; ++c) {
            p[c] = (shape == MORPH_RECT) ? 1 : ((r == cy || c == cx) ? 1 : 0);
        }
    }
    return k;
}

namespace {

template <bool Erode> Mat morphImpl(const Mat &src, const Mat &kernel) {
    MYCV_ASSERT(!src.empty());
    MYCV_ASSERT(!kernel.empty());
    MYCV_ASSERT(depthOf(src.type) == DEPTH_8U);
    MYCV_ASSERT(depthOf(kernel.type) == DEPTH_8U);
    MYCV_ASSERT(channels(kernel.type) == 1);

    const int ax = kernel.cols / 2;
    const int ay = kernel.rows / 2;
    const int cn = channels(src.type);

    Mat dst;
    dst.create(src.rows, src.cols, src.type);

    for (int y = 0; y < src.rows; ++y) {
        uint8_t *d = dst.ptr(y);
        for (int x = 0; x < src.cols; ++x) {
            for (int c = 0; c < cn; ++c) {
                int v = Erode ? 255 : 0; // erode 从最大开始找最小，dilate 反之
                for (int ky = 0; ky < kernel.rows; ++ky) {
                    const int sy = y + ky - ay;
                    if (sy < 0 || sy >= src.rows) continue; // 越界直接跳过，不反射
                    const uint8_t *krow = kernel.ptr(ky);
                    const uint8_t *srow = src.ptr(sy);
                    for (int kx = 0; kx < kernel.cols; ++kx) {
                        if (krow[kx] == 0) continue; // 核里是 0 的位置不参与
                        const int sx = x + kx - ax;
                        if (sx < 0 || sx >= src.cols) continue;
                        const int p = srow[sx * cn + c];
                        v = Erode ? std::min(v, p) : std::max(v, p);
                    }
                }
                d[x * cn + c] = static_cast<uint8_t>(v);
            }
        }
    }
    return dst;
}

} // namespace

Mat erode(const Mat &src, const Mat &kernel) { return morphImpl<true>(src, kernel); }
Mat dilate(const Mat &src, const Mat &kernel) { return morphImpl<false>(src, kernel); }

} // namespace mycv