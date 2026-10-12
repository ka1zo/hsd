#include "hsd/lexical_index.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_rank_and_metadata() {
    hsd::LexicalIndex index;
    index.add_chunks({{"a.txt", "apple apple banana", 4, 22},
                      {"b.md", "apple banana", 7, 19},
                      {"c.txt", "banana carrot", 0, 13}});

    const auto results = index.search("APPLE!");
    require(index.size() == 3, "wrong index size");
    require(results.size() == 2, "unexpected candidate count");
    require(results[0].chunk.source_path == "a.txt", "term frequency did not affect rank");
    require(results[0].chunk.start_byte == 4 && results[0].chunk.end_byte == 22,
            "source positions changed");
    require(results[0].chunk.text == "apple apple banana", "result text changed");
    require(results[0].lexical_score > results[1].lexical_score &&
                results[1].lexical_score > 0.0,
            "invalid lexical scores");

    const auto repeated = index.search("apple apple");
    require(repeated.size() == results.size(), "query repetition changed candidate count");
    require(std::abs(repeated[0].lexical_score - results[0].lexical_score) < 1e-12,
            "repeated query word inflated score");
    require(index.search("apple", 1).size() == 1, "limit was ignored");
    require(index.search("apple", 0).empty(), "zero limit was ignored");
    require(index.search("pear").empty(), "missing word matched");
    require(index.search("!? \n").empty(), "punctuation query matched");
    require(index.search("apple carrot").size() == 3, "multiword query did not union matches");
}

void test_russian_and_ties() {
    hsd::LexicalIndex index;
    index.add_chunks({{"z.md", "Привет, мир!", 10, 31},
                      {"a.txt", "привет мир", 0, 19}});
    const auto results = index.search("ПРИВЕТ");
    require(results.size() == 2, "Russian case folding failed");
    require(results[0].chunk.source_path == "a.txt", "equal-score tie order was unstable");
    require(results[1].chunk.source_path == "z.md", "second tie result missing");
}

void test_invalid_batch_does_not_change_index() {
    hsd::LexicalIndex index;
    index.add_chunks({{"existing.txt", "ready", 0, 5}});
    bool rejected = false;
    try {
        index.add_chunks({{"new.txt", "newword", 0, 7}, {"", "bad", 0, 3}});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid chunk was accepted");
    require(index.size() == 1 && index.search("newword").empty(),
            "invalid batch changed the index");

    rejected = false;
    try {
        index.add_chunks({{"bad.txt", std::string("\x80"), 0, 1}});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected && index.size() == 1, "malformed UTF-8 changed index");
}

void test_invalid_query_is_rejected_even_when_empty() {
    hsd::LexicalIndex index;
    bool rejected = false;
    try {
        (void)index.search(std::string("\x80"));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "malformed UTF-8 query was accepted");
}

} // namespace

int main() {
    test_rank_and_metadata();
    test_russian_and_ties();
    test_invalid_batch_does_not_change_index();
    test_invalid_query_is_rejected_even_when_empty();
}
