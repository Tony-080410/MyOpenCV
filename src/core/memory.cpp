#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include <mycv/core/memory.hpp>

namespace mycv {
std::shared_ptr<uint8_t> makeBuffer(size_t bytes) {
    if (bytes == 0) {
        return {};
    }
    void *p = ::operator new(bytes, std::align_val_t{64});
    return std::shared_ptr<uint8_t>(static_cast<uint8_t *>(p),
                                    [](uint8_t *q) { ::operator delete(q, std::align_val_t{64}); });
}
} // namespace mycv