#include "hsd/hybrid_index.hpp"

#include <algorithm>
#include <numeric>

namespace hsd {

void HybridIndex::add_chunks(const std::vector<Chunk>& chunks,
                             const std::vector<std::vector<float>>& embeddings) {
    // Validate both complete batches before changing either persistent index.
    LexicalIndex lexical_batch;
    VectorIndex vector_batch;
    lexical_batch.add_chunks(chunks);
    vector_batch.add_chunks(chunks, embeddings);
    lexical_.add_chunks(chunks);
    vector_.add_chunks(chunks, embeddings);
}

std::vector<HybridSearchResult> HybridIndex::search(
    std::string_view query, const std::vector<float>& query_embedding,
    std::size_t limit) const {
    const auto lexical = lexical_.search(query, lexical_.size());
    const auto vector = vector_.search(query_embedding, vector_.size());
    if (limit == 0 || vector_.size() == 0) return {};

    constexpr double rank_constant = 60.0;
    std::vector<HybridSearchResult> scores(vector_.size());
    for (std::size_t rank = 0; rank < vector.size(); ++rank) {
        auto& result = scores[vector[rank].chunk_index];
        result.chunk = vector[rank].chunk;
        result.vector_score = vector[rank].vector_score;
        result.hybrid_score += 1.0 / (rank_constant + rank + 1);
    }
    for (std::size_t rank = 0; rank < lexical.size(); ++rank) {
        auto& result = scores[lexical[rank].chunk_index];
        result.lexical_score = lexical[rank].lexical_score;
        result.hybrid_score += 1.0 / (rank_constant + rank + 1);
    }

    std::vector<std::size_t> order(vector_.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        if (scores[left].hybrid_score != scores[right].hybrid_score) {
            return scores[left].hybrid_score > scores[right].hybrid_score;
        }
        const auto& a = scores[left].chunk;
        const auto& b = scores[right].chunk;
        if (a.source_path != b.source_path) return a.source_path < b.source_path;
        if (a.start_byte != b.start_byte) return a.start_byte < b.start_byte;
        if (a.end_byte != b.end_byte) return a.end_byte < b.end_byte;
        return left < right;
    });
    std::vector<HybridSearchResult> results;
    results.reserve(std::min(limit, order.size()));
    for (std::size_t rank = 0; rank < std::min(limit, order.size()); ++rank) {
        results.push_back(std::move(scores[order[rank]]));
    }
    return results;
}

std::size_t HybridIndex::size() const noexcept { return vector_.size(); }

} // namespace hsd
