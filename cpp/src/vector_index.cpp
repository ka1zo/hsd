#include "hsd/vector_index.hpp"

#include "hsd/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace hsd {
namespace {

void validate_vector(const std::vector<float>& vector) {
    if (vector.size() != VectorIndex::dimensions) {
        throw std::invalid_argument("invalid embedding dimension");
    }
    double squared_norm = 0.0;
    for (float value : vector) {
        if (!std::isfinite(value)) throw std::invalid_argument("invalid embedding value");
        squared_norm += static_cast<double>(value) * value;
    }
    if (std::abs(squared_norm - 1.0) > 0.001) {
        throw std::invalid_argument("embedding must be normalized");
    }
}

} // namespace

void VectorIndex::add_chunks(const std::vector<Chunk>& chunks,
                             const std::vector<std::vector<float>>& embeddings) {
    if (chunks.size() != embeddings.size()) {
        throw std::invalid_argument("chunk and embedding counts differ");
    }
    for (std::size_t index = 0; index < chunks.size(); ++index) {
        if (!is_valid(chunks[index])) throw std::invalid_argument("invalid chunk");
        (void)tokenize(chunks[index].source_path);
        (void)tokenize(chunks[index].text);
        validate_vector(embeddings[index]);
    }
    chunks_.insert(chunks_.end(), chunks.begin(), chunks.end());
    embeddings_.insert(embeddings_.end(), embeddings.begin(), embeddings.end());
}

std::vector<VectorSearchResult> VectorIndex::search(const std::vector<float>& query,
                                                    std::size_t limit) const {
    validate_vector(query);
    if (limit == 0 || chunks_.empty()) return {};

    std::vector<std::pair<std::size_t, double>> ranked;
    ranked.reserve(chunks_.size());
    for (std::size_t index = 0; index < embeddings_.size(); ++index) {
        double score = 0.0;
        for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
            score += static_cast<double>(query[dimension]) * embeddings_[index][dimension];
        }
        ranked.emplace_back(index, score);
    }
    std::sort(ranked.begin(), ranked.end(), [this](const auto& left, const auto& right) {
        if (left.second != right.second) return left.second > right.second;
        const auto& a = chunks_[left.first];
        const auto& b = chunks_[right.first];
        if (a.source_path != b.source_path) return a.source_path < b.source_path;
        if (a.start_byte != b.start_byte) return a.start_byte < b.start_byte;
        if (a.end_byte != b.end_byte) return a.end_byte < b.end_byte;
        return left.first < right.first;
    });

    std::vector<VectorSearchResult> results;
    results.reserve(std::min(limit, ranked.size()));
    for (std::size_t index = 0; index < std::min(limit, ranked.size()); ++index) {
        results.push_back({chunks_[ranked[index].first], ranked[index].second,
                           ranked[index].first});
    }
    return results;
}

std::size_t VectorIndex::size() const noexcept { return chunks_.size(); }

} // namespace hsd
