#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mycv/core/check.hpp>
#include <mycv/core/mat.hpp>
#include <mycv/core/types.hpp>
#include <mycv/imgproc/geom.hpp>
#include <stdexcept>
#include <vector>
namespace mycv {
namespace {
// 权重精度 11 位，抄袭 OpenCV 的 INTER_RESIZE_COEF_BITS 。。。
constexpr int COEF_BITS = 11;
constexpr int COEF_SCALE = 1 << COEF_BITS; // 2048
constexpr int ACC_SHIFT = COEF_BITS * 2; // 22

// 一维的起点索引&2个权重：
// 规定半像素中心：(d+0.5)*scale-0.5
// 预先算出缩放所需的映射表,
// 一次性算好：每个输出位置对应源图哪两个像素、各占多少权重，从而避免在逐像素循环里反复做浮点运算。
void buildTable(int dstN, int srcN, double scale, std::vector<int> &ofs, std::vector<uint16_t> &w) {
    ofs.resize(dstN);
    w.resize(static_cast<size_t>(dstN) * 2);

    for (int d = 0; d < dstN; ++d) {
        double f = (d + 0.5) * scale - 0.5;
        int s = static_cast<int>(std::floor(f));
        f -= s;

        if (s < 0) {
            // 左/上边界情况，舍掉
            s = 0;
            f = 0.0;
        }
        if (s >= srcN - 1) {
            s = srcN - 1;
            f = 0.0;
        }

        ofs[d] = s;
        const int w1 = static_cast<int>(std::lround(f * COEF_SCALE));

        // 构造权重和为2048
        w[d * 2 + 1] = static_cast<uint16_t>(w1);
        w[d * 2 + 0] = static_cast<uint16_t>(COEF_SCALE - w1);
    }
}

} // namespace

Mat resize(const Mat &src, Size dstSize, int interpolation) {

    if (interpolation != INTER_LINEAR) {
        throw std::runtime_error("resze: 目前只支持INTER_LINEAR");
    }
    MYCV_ASSERT(!src.empty());
    MYCV_ASSERT(dstSize.width > 0 && dstSize.height > 0);
    MYCV_ASSERT(src.type == U8C1 || src.type == U8C3);

    const int cn = channels(src.type);
    const int dn = dstSize.width;
    const int dm = dstSize.height;

    const double scaleX = static_cast<double>(src.cols) / dn;
    const double scaleY = static_cast<double>(src.rows) / dm;

    std::vector<int> xofs, yofs;
    std::vector<uint16_t> xw, yw;
    buildTable(dn, src.cols, scaleX, xofs, xw);
    buildTable(dm, src.rows, scaleY, yofs, yw);

    Mat dst;
    dst.create(dm, dn, src.type);

    for (int dy = 0; dy < dm; ++dy) {
        const uint8_t *r0 = src.ptr(yofs[dy]);
        const uint8_t *r1 = src.ptr(std::min(yofs[dy] + 1, src.rows - 1));
        uint8_t *d = dst.ptr(dy);
        const int b0 = yw[dy * 2 + 0];
        const int b1 = yw[dy * 2 + 1];

        for (int dx = 0; dx < dn; ++dx) {
            const int x0 = xofs[dx] * cn;
            const int x1 = std::min(xofs[dx] + 1, src.cols - 1) * cn;
            const int a0 = xw[dx * 2 + 0];
            const int a1 = xw[dx * 2 + 1];

            for (int k = 0; k < cn; ++k) {
                // 权重和 = 2048*2048，最大累加 255*2048*2048 < INT_MAX，不会溢出。太棒了👍
                const int acc = r0[x0 + k] * a0 * b0 + r0[x1 + k] * a1 * b0 + r1[x0 + k] * a0 * b1 +
                                r1[x1 + k] * a1 * b1;
                d[dx * cn + k] = static_cast<uint8_t>((acc + (1 << (ACC_SHIFT - 1))) >> ACC_SHIFT);
            }
        }
    }
    return dst;
}
} // namespace mycv