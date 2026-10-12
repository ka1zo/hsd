#pragma once

#include <cstdint>
#include <string>

namespace hsd {

// Byte offsets refer to the UTF-8 representation of the source text.
struct Chunk {
    std::string source_path;
    std::string text;
    std::uint64_t start_byte;
    std::uint64_t end_byte;
};

[[nodiscard]] bool is_valid(const Chunk& chunk) noexcept;

} // namespace hsd
