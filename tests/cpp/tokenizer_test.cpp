#include "hsd/tokenizer.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    const std::vector<std::string> expected{"hello", "hello", "привет", "ёж", "café", "café"};
    require(hsd::tokenize("Hello, HELLO! ПрИвЕт ЁЖ café CAFÉ") == expected,
            "case folding or punctuation failed");
    require(hsd::tokenize(" \n!? ").empty(), "punctuation should yield no words");

    for (const std::string& invalid : {std::string("\xC0\xAF"), std::string("\xD0"), std::string("\x80")}) {
        bool rejected = false;
        try {
            (void)hsd::tokenize(invalid);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "malformed UTF-8 was accepted");
    }
}
