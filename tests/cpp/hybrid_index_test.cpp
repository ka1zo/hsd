#include "hsd/hybrid_index.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<float> unit(std::size_t coordinate) {
    std::vector<float> vector(384, 0.0f);
    vector[coordinate] = 1.0f;
    return vector;
}

void test_fusion_keeps_both_scores_and_positions() {
    hsd::HybridIndex index;
    index.add_chunks({{"a.txt", "apple", 3, 8},
                      {"b.txt", "banana", 0, 6},
                      {"c.txt", "apple", 10, 15}},
                     {unit(1), unit(0), unit(2)});
    const auto results = index.search("apple", unit(0));
    require(index.size() == 3 && results.size() == 3, "candidate union is incomplete");
    require(results[0].chunk.source_path == "a.txt" && results[0].chunk.start_byte == 3,
            "lexical and vector ranks were not fused");
    require(results[0].lexical_score > 0 && results[0].vector_score == 0,
            "source scores were lost");
    require(results[1].chunk.source_path == "c.txt" && results[1].lexical_score > 0,
            "second lexical candidate missing");
    require(results[2].chunk.source_path == "b.txt" && results[2].lexical_score == 0 &&
                results[2].vector_score == 1,
            "vector-only candidate missing");
    require(std::abs(results[0].hybrid_score - (1.0 / 61.0 + 1.0 / 62.0)) < 1e-12,
            "RRF score is wrong");
    require(index.search("apple", unit(0), 1).size() == 1, "limit ignored");
    require(index.search("apple", unit(0), 0).empty(), "zero limit ignored");
}

void test_duplicate_fragments_remain_distinct() {
    hsd::HybridIndex index;
    const hsd::Chunk same{"a.txt", "apple", 0, 5};
    index.add_chunks({same, same}, {unit(0), unit(1)});
    const auto results = index.search("apple", unit(0));
    require(results.size() == 2 && results[0].vector_score == 1.0 &&
                results[1].vector_score == 0.0,
            "duplicate fragments were conflated");
}

void test_invalid_batch_preserves_index() {
    hsd::HybridIndex index;
    index.add_chunks({{"original", "apple", 0, 5}}, {unit(0)});
    bool rejected = false;
    try {
        index.add_chunks({{"new", "new", 0, 3}}, {std::vector<float>(383)});
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && index.size() == 1, "invalid vector changed hybrid index");
    rejected = false;
    try {
        index.add_chunks({{"", "bad", 0, 3}}, {unit(0)});
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && index.size() == 1, "invalid fragment changed hybrid index");
    rejected = false;
    try { (void)index.search("apple", std::vector<float>(383)); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid query vector accepted");
}

} // namespace

int main() {
    test_fusion_keeps_both_scores_and_positions();
    test_duplicate_fragments_remain_distinct();
    test_invalid_batch_preserves_index();
}
