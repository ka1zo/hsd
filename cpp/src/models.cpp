#include "hsd/models.hpp"

namespace hsd {

bool is_valid(const Chunk& chunk) noexcept {
    return !chunk.source_path.empty() && !chunk.text.empty() &&
           chunk.start_byte < chunk.end_byte;
}

} // namespace hsd
