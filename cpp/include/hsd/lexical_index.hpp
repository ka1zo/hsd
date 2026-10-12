#pragma once

#include "hsd/models.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hsd {

struct SearchResult {
    Chunk chunk;
    double lexical_score;
    std::size_t chunk_index;
};

class LexicalIndex {
public:
    void add_chunks(const std::vector<Chunk>& chunks);
    [[nodiscard]] std::vector<SearchResult> search(std::string_view query,
                                                   std::size_t limit = 10) const;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    struct Posting {
        std::size_t chunk_index;
        std::size_t frequency;
    };

    std::vector<Chunk> chunks_;
    std::vector<std::size_t> lengths_;
    std::unordered_map<std::string, std::vector<Posting>> postings_;
    std::size_t total_tokens_ = 0;
};

} // namespace hsd
