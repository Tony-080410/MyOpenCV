#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

namespace mycv {
std::shared_ptr<uint8_t> makeBuffer(std::size_t bytes);
} // namespace mycv
