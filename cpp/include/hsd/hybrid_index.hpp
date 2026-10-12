#pragma once

#include "hsd/lexical_index.hpp"
#include "hsd/vector_index.hpp"

#include <cstddef>
#include <string_view>
#include <vector>

namespace hsd {

struct HybridSearchResult {
    Chunk chunk;
    double lexical_score;
    double vector_score;
    double hybrid_score;
};

class HybridIndex {
public:
    void add_chunks(const std::vector<Chunk>& chunks,
                    const std::vector<std::vector<float>>& embeddings);
    [[nodiscard]] std::vector<HybridSearchResult> search(
        std::string_view query, const std::vector<float>& query_embedding,
        std::size_t limit = 10) const;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    LexicalIndex lexical_;
    VectorIndex vector_;
};

} // namespace hsd
