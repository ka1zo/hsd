#include "hsd/tokenizer.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace hsd {
namespace {

std::uint32_t next_codepoint(std::string_view text, std::size_t& offset) {
    const auto first = static_cast<unsigned char>(text[offset++]);
    if (first < 0x80) return first;

    int continuation_count = 0;
    std::uint32_t codepoint = 0;
    std::uint32_t minimum = 0;
    if (first >= 0xC2 && first <= 0xDF) {
        continuation_count = 1;
        codepoint = first & 0x1F;
        minimum = 0x80;
    } else if (first >= 0xE0 && first <= 0xEF) {
        continuation_count = 2;
        codepoint = first & 0x0F;
        minimum = 0x800;
    } else if (first >= 0xF0 && first <= 0xF4) {
        continuation_count = 3;
        codepoint = first & 0x07;
        minimum = 0x10000;
    } else {
        throw std::invalid_argument("invalid UTF-8");
    }

    if (text.size() - offset < static_cast<std::size_t>(continuation_count)) {
        throw std::invalid_argument("invalid UTF-8");
    }
    for (int index = 0; index < continuation_count; ++index) {
        const auto byte = static_cast<unsigned char>(text[offset++]);
        if ((byte & 0xC0) != 0x80) throw std::invalid_argument("invalid UTF-8");
        codepoint = (codepoint << 6) | (byte & 0x3F);
    }
    if (codepoint < minimum || codepoint > 0x10FFFF ||
        (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
        throw std::invalid_argument("invalid UTF-8");
    }
    return codepoint;
}

bool is_word_codepoint(std::uint32_t cp) {
    return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') ||
           (cp >= '0' && cp <= '9') ||
           (cp >= 0xC0 && cp <= 0xD6) || (cp >= 0xD8 && cp <= 0xF6) ||
           (cp >= 0xF8 && cp <= 0xFF) ||
           (cp >= 0x0400 && cp <= 0x045F);
}

std::uint32_t lowercase(std::uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 0x20;
    if ((cp >= 0xC0 && cp <= 0xD6) || (cp >= 0xD8 && cp <= 0xDE)) return cp + 0x20;
    if (cp >= 0x0400 && cp <= 0x040F) return cp + 0x50;
    if (cp >= 0x0410 && cp <= 0x042F) return cp + 0x20;
    return cp;
}

void append_utf8(std::string& output, std::uint32_t cp) {
    if (cp < 0x80) {
        output.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        output.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        output.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        output.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        output.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

} // namespace

std::vector<std::string> tokenize(std::string_view text) {
    std::vector<std::string> words;
    std::string current;
    std::size_t offset = 0;
    while (offset < text.size()) {
        const auto codepoint = next_codepoint(text, offset);
        if (is_word_codepoint(codepoint)) {
            append_utf8(current, lowercase(codepoint));
        } else if (!current.empty()) {
            words.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) words.push_back(std::move(current));
    return words;
}

} // namespace hsd
