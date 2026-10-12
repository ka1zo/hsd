#include "hsd/lexical_index.hpp"

#include "hsd/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace hsd {

void LexicalIndex::add_chunks(const std::vector<Chunk>& chunks) {
    std::vector<std::unordered_map<std::string, std::size_t>> frequencies;
    std::vector<std::size_t> lengths;
    frequencies.reserve(chunks.size());
    lengths.reserve(chunks.size());

    // Validate the complete batch before changing the index.
    for (const auto& chunk : chunks) {
        if (!is_valid(chunk)) throw std::invalid_argument("invalid chunk");
        auto words = tokenize(chunk.text);
        lengths.push_back(words.size());
        std::unordered_map<std::string, std::size_t> counts;
        for (const auto& word : words) ++counts[word];
        frequencies.push_back(std::move(counts));
    }

    for (std::size_t index = 0; index < chunks.size(); ++index) {
        const auto chunk_index = chunks_.size();
        chunks_.push_back(chunks[index]);
        lengths_.push_back(lengths[index]);
        total_tokens_ += lengths[index];
        for (const auto& [term, frequency] : frequencies[index]) {
            postings_[term].push_back({chunk_index, frequency});
        }
    }
}

std::vector<SearchResult> LexicalIndex::search(std::string_view query, std::size_t limit) const {
    auto words = tokenize(query);
    if (limit == 0 || chunks_.empty() || words.empty()) return {};
    std::sort(words.begin(), words.end());
    words.erase(std::unique(words.begin(), words.end()), words.end());

    constexpr double k1 = 1.2;
    constexpr double b = 0.75;
    const double average_length = static_cast<double>(total_tokens_) / chunks_.size();
    std::vector<double> scores(chunks_.size(), 0.0);
    std::vector<bool> matched(chunks_.size(), false);

    for (const auto& word : words) {
        const auto found = postings_.find(word);
        if (found == postings_.end()) continue;
        const auto& list = found->second;
        const double document_count = static_cast<double>(chunks_.size());
        const double document_frequency = static_cast<double>(list.size());
        const double idf = std::log1p((document_count - document_frequency + 0.5) /
                                      (document_frequency + 0.5));
        for (const auto& posting : list) {
            const double frequency = static_cast<double>(posting.frequency);
            const double length = static_cast<double>(lengths_[posting.chunk_index]);
            const double normalization = k1 * (1.0 - b + b * length / average_length);
            scores[posting.chunk_index] += idf * frequency * (k1 + 1.0) /
                                           (frequency + normalization);
            matched[posting.chunk_index] = true;
        }
    }

    std::vector<std::pair<std::size_t, double>> ranked;
    for (std::size_t index = 0; index < chunks_.size(); ++index) {
        if (matched[index]) ranked.emplace_back(index, scores[index]);
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

    std::vector<SearchResult> results;
    results.reserve(std::min(limit, ranked.size()));
    for (std::size_t index = 0; index < std::min(limit, ranked.size()); ++index) {
        results.push_back({chunks_[ranked[index].first], ranked[index].second,
                           ranked[index].first});
    }
    return results;
}

std::size_t LexicalIndex::size() const noexcept { return chunks_.size(); }

} // namespace hsd
