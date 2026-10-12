#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace hsd {

// Supports ASCII, Latin-1 letters and Russian Cyrillic letters/digits.
// Malformed UTF-8 raises std::invalid_argument.
[[nodiscard]] std::vector<std::string> tokenize(std::string_view text);

} // namespace hsd
