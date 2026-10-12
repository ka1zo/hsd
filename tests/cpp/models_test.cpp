#include "hsd/models.hpp"

int main() {
    const hsd::Chunk valid{"notes/a.md", "hello", 0, 5};
    if (!hsd::is_valid(valid)) return 1;
    if (hsd::is_valid({"", "hello", 0, 5})) return 2;
    if (hsd::is_valid({"notes/a.md", "", 0, 5})) return 3;
    if (hsd::is_valid({"notes/a.md", "hello", 5, 5})) return 4;
    if (hsd::is_valid({"notes/a.md", "hello", 6, 5})) return 5;
}
