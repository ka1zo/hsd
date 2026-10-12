#include "hsd/vector_index.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<float> axis(std::size_t coordinate, float sign = 1.0f) {
    std::vector<float> vector(384, 0.0f);
    vector[coordinate] = sign;
    return vector;
}

void test_exact_scores_and_ties() {
    hsd::VectorIndex index;
    index.add_chunks({{"z.txt", "opposite", 0, 8},
                      {"b.txt", "related", 3, 10},
                      {"a.txt", "same", 7, 11},
                      {"a.txt", "same again", 15, 25}},
                     {axis(0, -1), axis(1), axis(0), axis(0)});
    const auto results = index.search(axis(0));
    require(index.size() == 4 && results.size() == 4, "missing vector candidates");
    require(results[0].chunk.start_byte == 7 && results[1].chunk.start_byte == 15,
            "equal-score tie order was unstable");
    require(results[0].vector_score == 1.0 && results[1].vector_score == 1.0,
            "positive cosine score is wrong");
    require(results[2].chunk.source_path == "b.txt" && results[2].vector_score == 0.0,
            "orthogonal score is wrong");
    require(results[3].vector_score == -1.0, "negative score was lost");
    require(index.search(axis(0), 2).size() == 2, "limit was ignored");
    require(index.search(axis(0), 0).empty(), "zero limit was ignored");
}

void test_invalid_vectors_do_not_change_index() {
    hsd::VectorIndex index;
    index.add_chunks({{"original", "original", 0, 8}}, {axis(0)});
    auto reject_batch = [&](const std::vector<hsd::Chunk>& chunks,
                            const std::vector<std::vector<float>>& vectors) {
        bool rejected = false;
        try { index.add_chunks(chunks, vectors); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && index.size() == 1, "invalid batch changed index");
    };
    reject_batch({{"a", "new", 0, 3}}, {});
    reject_batch({{"a", "new", 0, 3}}, {std::vector<float>(383, 0.0f)});
    reject_batch({{"a", "new", 0, 3}}, {axis(0, 2)});
    auto nan = axis(0);
    nan[0] = std::nanf("");
    reject_batch({{"a", "new", 0, 3}}, {nan});
    reject_batch({{"a", "new", 0, 3}, {"", "bad", 0, 3}}, {axis(0), axis(1)});
    require(index.search(axis(0))[0].chunk.source_path == "original", "original vector lost");

    for (const auto& bad : {std::vector<float>(383, 0.0f), axis(0, 2), nan}) {
        bool rejected = false;
        try { (void)index.search(bad); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid query vector accepted");
    }
}

} // namespace

int main() {
    test_exact_scores_and_ties();
    test_invalid_vectors_do_not_change_index();
}
