#include <cstddef>
#include <cstring>
#include <mycv/core/check.hpp>
#include <mycv/core/mat.hpp>
#include <mycv/core/memory.hpp>
#include <mycv/core/types.hpp>
namespace mycv {
void Mat::create(int r, int c, int t) {
    MYCV_ASSERT(r > 0 && c > 0);
    MYCV_ASSERT(elemSize(t) > 0);

    const size_t es = elemSize(t);
    const size_t bytes = static_cast<size_t>(r) * static_cast<size_t>(c) * es;

    auto buf = makeBuffer(bytes);
    owner = buf;
    data = owner.get();
    rows = r;
    cols = c;
    type = t;
    step = static_cast<size_t>(c) * es;
}
uint8_t *Mat::ptr(int r) {
    MYCV_ASSERT(data != nullptr);
    MYCV_ASSERT(r >= 0 && r < rows);
    return data + static_cast<std::size_t>(r) * step;
}

const uint8_t *Mat::ptr(int r) const {
    MYCV_ASSERT(data != nullptr);
    MYCV_ASSERT(r >= 0 && r < rows);
    return data + static_cast<std::size_t>(r) * step;
}

Mat Mat::clone() const {
    MYCV_ASSERT(!empty());
    Mat out;
    out.create(rows, cols, type);

    const size_t rowBytes = static_cast<size_t>(cols) * elemSize(type);
    for (int i = 0; i < rows; ++i) {
        memcpy(out.ptr(i), ptr(i), rowBytes);
    }

    return out;
}

Mat Mat::roi(const Rect &r) const {
    MYCV_ASSERT(!empty());
    MYCV_ASSERT(r.x >= 0 && r.y >= 0 && r.width > 0 && r.height > 0);
    MYCV_ASSERT(r.x + r.width <= cols && r.y + r.height <= rows);

    Mat out;
    out.rows = r.height;
    out.cols = r.width;
    out.type = type;
    out.step = step;
    out.owner = owner;
    out.data = data + static_cast<size_t>(r.y) * step + static_cast<size_t>(r.x) * elemSize(type);
    return out;
}

void Mat::release() {
    owner.reset();
    data = nullptr;
    rows = 0;
    cols = 0;
    type = 0;
    step = 0;
}

} // namespace mycv