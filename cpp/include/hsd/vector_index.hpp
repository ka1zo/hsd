#pragma once

#include "hsd/models.hpp"

#include <cstddef>
#include <vector>

namespace hsd {

struct VectorSearchResult {
    Chunk chunk;
    double vector_score;
    std::size_t chunk_index;
};

class VectorIndex {
public:
    static constexpr std::size_t dimensions = 384;

    void add_chunks(const std::vector<Chunk>& chunks,
                    const std::vector<std::vector<float>>& embeddings);
    [[nodiscard]] std::vector<VectorSearchResult> search(const std::vector<float>& query,
                                                          std::size_t limit = 10) const;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    std::vector<Chunk> chunks_;
    std::vector<std::vector<float>> embeddings_;
};

} // namespace hsd
