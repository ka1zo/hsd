#pragma once

#include "hsd/models.hpp"

#include <stdexcept>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct sqlite3;

namespace hsd {

class StoreError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class EncryptedStore {
public:
    EncryptedStore(const std::string& path, const std::string& password, bool create);
    ~EncryptedStore();
    EncryptedStore(const EncryptedStore&) = delete;
    EncryptedStore& operator=(const EncryptedStore&) = delete;

    void replace_chunks(const std::vector<Chunk>& chunks);
    void replace_chunks_with_embeddings(const std::vector<Chunk>& chunks,
                                        const std::vector<std::vector<float>>& embeddings,
                                        const std::string& model_id);
    [[nodiscard]] std::vector<Chunk> read_chunks() const;
    [[nodiscard]] std::pair<std::string, std::vector<std::vector<float>>> read_embeddings() const;
    [[nodiscard]] std::pair<std::string, std::map<std::string, std::string>> read_file_hashes() const;
    [[nodiscard]] std::string read_model_id() const;
    [[nodiscard]] std::size_t apply_update(
        const std::string& root,
        const std::vector<std::pair<std::string, std::string>>& files,
        const std::vector<Chunk>& chunks,
        const std::vector<std::vector<float>>& embeddings,
        const std::string& model_id,
        const std::vector<std::string>& prepared_paths);

private:
    sqlite3* db_ = nullptr;
    int version_ = 0;
};

} // namespace hsd
