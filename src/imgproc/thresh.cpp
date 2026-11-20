#include "mycv/core/check.hpp"
#include "mycv/core/types.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mycv/core/mat.hpp>
#include <mycv/imgproc/thresh.hpp>

#include <stdexcept>
namespace mycv {
Mat threshold(const Mat &src, double thresh, double maxval, int type) {

    MYCV_ASSERT(!src.empty());
    MYCV_ASSERT(depthOf(src.type) == DEPTH_8U);

    if (type < THRESH_BINARY || type > THRESH_TOZERO_INV) {
        throw std::runtime_error("threshold: 未知或未实现的type");
    }

    const int t = static_cast<int>(std::floor(thresh));
    int mv = static_cast<int>(std::lround(maxval));

    if (mv > 255) {
        mv = 255;
    }
    if (mv < 0) {
        mv = 0;
    }
    Mat dst = src.clone();
    uint8_t *p = dst.data;
    const size_t n = static_cast<size_t>((dst.rows) * dst.step);

    for (size_t i = 0; i < n; ++i) {
        const int v = p[i];
        switch (type) {
        case THRESH_BINARY:
            p[i] = static_cast<uint8_t>(v > t ? mv : 0);
            break;
        case THRESH_BINARY_INV:
            p[i] = static_cast<uint8_t>(v > t ? 0 : mv);
            break;
        case THRESH_TRUNC: {
            int r = (v > t) ? t : v;
            if (r < 0) r = 0;
            if (r > 255) r = 255;
            p[i] = static_cast<uint8_t>(r);
            break;
        }
        case THRESH_TOZERO:
            p[i] = static_cast<uint8_t>(v > t ? v : 0);
            break;

        case THRESH_TOZERO_INV:
            p[i] = static_cast<uint8_t>(v > t ? 0 : v);
            break;
        default:
            break;
        }
    }
    return dst;
}
} // namespace mycv