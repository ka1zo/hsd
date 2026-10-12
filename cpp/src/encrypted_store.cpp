#include "hsd/encrypted_store.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>

namespace hsd {
namespace {

constexpr int schema_version = 3;
constexpr std::size_t embedding_dimensions = 384;

bool valid_digest(const std::string& digest) {
    return digest.size() == 64 && std::all_of(digest.begin(), digest.end(), [](char value) {
        return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
    });
}

void exec(sqlite3* db, const char* sql) {
    if (sqlite3_exec(db, sql, nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw StoreError("Encrypted index operation failed.");
    }
}

class Statement {
public:
    Statement(sqlite3* db, const char* sql) {
        if (sqlite3_prepare_v2(db, sql, -1, &value_, nullptr) != SQLITE_OK) {
            throw StoreError("Encrypted index operation failed.");
        }
    }
    ~Statement() { sqlite3_finalize(value_); }
    sqlite3_stmt* get() const { return value_; }
private:
    sqlite3_stmt* value_ = nullptr;
};

int scalar_int(sqlite3* db, const char* sql) {
    Statement statement(db, sql);
    if (sqlite3_step(statement.get()) != SQLITE_ROW) {
        throw StoreError("Encrypted index is damaged or the password is incorrect.");
    }
    return sqlite3_column_int(statement.get(), 0);
}

} // namespace

EncryptedStore::EncryptedStore(const std::string& path, const std::string& password, bool create) {
    if (path.empty() || password.empty() || password.size() > std::numeric_limits<int>::max()) {
        throw StoreError("An index path and nonempty password are required.");
    }
    const int flags = create ? SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE : SQLITE_OPEN_READONLY;
    if (sqlite3_open_v2(path.c_str(), &db_, flags | SQLITE_OPEN_NOMUTEX, nullptr) != SQLITE_OK) {
        sqlite3_close(db_);
        db_ = nullptr;
        throw StoreError("Cannot open encrypted index.");
    }
    try {
        if (sqlite3_key(db_, password.data(), static_cast<int>(password.size())) != SQLITE_OK) {
            throw StoreError("Cannot unlock encrypted index.");
        }
        Statement cipher(db_, "PRAGMA cipher_version");
        if (sqlite3_step(cipher.get()) != SQLITE_ROW || sqlite3_column_text(cipher.get(), 0) == nullptr ||
            reinterpret_cast<const char*>(sqlite3_column_text(cipher.get(), 0))[0] != '4' ||
            reinterpret_cast<const char*>(sqlite3_column_text(cipher.get(), 0))[1] != '.') {
            throw StoreError("SQLCipher 4.x is required to open an encrypted index.");
        }
        exec(db_, "PRAGMA temp_store=MEMORY");
        // This also proves that the password can decrypt an existing database.
        try {
            (void)scalar_int(db_, "SELECT count(*) FROM sqlite_master");
        } catch (const StoreError&) {
            throw StoreError("Encrypted index is damaged or the password is incorrect.");
        }
        {
            Statement integrity(db_, "PRAGMA cipher_integrity_check");
            if (sqlite3_step(integrity.get()) != SQLITE_DONE) {
                throw StoreError("Encrypted index is damaged.");
            }
        }
        const int version = scalar_int(db_, "PRAGMA user_version");
        if (version > schema_version) {
            throw StoreError("Encrypted index format is newer than this HSD version.");
        }
        if (version == 0) {
            if (!create || scalar_int(db_, "SELECT count(*) FROM sqlite_master") != 0) {
                throw StoreError("Encrypted index format is unsupported.");
            }
            exec(db_, "BEGIN IMMEDIATE");
            try {
                exec(db_, "CREATE TABLE chunks (id INTEGER PRIMARY KEY, source_path TEXT NOT NULL, text TEXT NOT NULL, start_byte INTEGER NOT NULL, end_byte INTEGER NOT NULL)");
                exec(db_, "CREATE TABLE embedding_model (id INTEGER PRIMARY KEY CHECK(id=1), model_id TEXT NOT NULL)");
                exec(db_, "CREATE TABLE embeddings (chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id), vector BLOB NOT NULL)");
                exec(db_, "CREATE TABLE index_state (id INTEGER PRIMARY KEY CHECK(id=1), root TEXT NOT NULL)");
                exec(db_, "INSERT INTO index_state(id,root) VALUES(1,'')");
                exec(db_, "CREATE TABLE indexed_files (path TEXT PRIMARY KEY, digest TEXT NOT NULL)");
                exec(db_, "CREATE INDEX chunks_source_path ON chunks(source_path)");
                exec(db_, "PRAGMA user_version=3");
                exec(db_, "COMMIT");
            } catch (...) {
                sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
                throw;
            }
        }
        if (version == 1 && create) {
            exec(db_, "BEGIN IMMEDIATE");
            try {
                exec(db_, "CREATE TABLE embedding_model (id INTEGER PRIMARY KEY CHECK(id=1), model_id TEXT NOT NULL)");
                exec(db_, "CREATE TABLE embeddings (chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id), vector BLOB NOT NULL)");
                exec(db_, "PRAGMA user_version=2");
                exec(db_, "COMMIT");
            } catch (...) {
                sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
                throw;
            }
        }
        if ((version == 1 || version == 2) && create) {
            exec(db_, "BEGIN IMMEDIATE");
            try {
                exec(db_, "CREATE TABLE index_state (id INTEGER PRIMARY KEY CHECK(id=1), root TEXT NOT NULL)");
                exec(db_, "INSERT INTO index_state(id,root) VALUES(1,'')");
                exec(db_, "CREATE TABLE indexed_files (path TEXT PRIMARY KEY, digest TEXT NOT NULL)");
                exec(db_, "CREATE INDEX chunks_source_path ON chunks(source_path)");
                exec(db_, "PRAGMA user_version=3");
                exec(db_, "COMMIT");
            } catch (...) {
                sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
                throw;
            }
        }
        version_ = create ? schema_version : version;
        if (create) {
            exec(db_, "PRAGMA secure_delete=ON");
            exec(db_, "PRAGMA journal_mode=DELETE");
        }
    } catch (const StoreError&) {
        sqlite3_close(db_);
        db_ = nullptr;
        throw;
    }
}

EncryptedStore::~EncryptedStore() { sqlite3_close(db_); }

void EncryptedStore::replace_chunks(const std::vector<Chunk>& chunks) {
    replace_chunks_with_embeddings(chunks, {}, "");
}

void EncryptedStore::replace_chunks_with_embeddings(const std::vector<Chunk>& chunks,
                                                     const std::vector<std::vector<float>>& embeddings,
                                                     const std::string& model_id) {
    if ((!model_id.empty() && embeddings.size() != chunks.size()) ||
        (model_id.empty() && !embeddings.empty()) || model_id.size() > std::numeric_limits<int>::max()) {
        throw StoreError("Invalid embedding metadata.");
    }
    for (const auto& vector : embeddings) {
        if (vector.size() != embedding_dimensions) {
            throw StoreError("Invalid embedding dimension.");
        }
        double squared_norm = 0;
        for (float value : vector) {
            if (!std::isfinite(value)) {
                throw StoreError("Invalid embedding value.");
            }
            squared_norm += static_cast<double>(value) * value;
        }
        if (std::abs(squared_norm - 1.0) > 0.001) {
            throw StoreError("Embedding must be normalized.");
        }
    }
    for (const auto& chunk : chunks) {
        if (!is_valid(chunk) || chunk.source_path.size() > std::numeric_limits<int>::max() ||
            chunk.text.size() > std::numeric_limits<int>::max() ||
            chunk.start_byte > static_cast<std::uint64_t>(std::numeric_limits<sqlite3_int64>::max()) ||
            chunk.end_byte > static_cast<std::uint64_t>(std::numeric_limits<sqlite3_int64>::max())) {
            throw StoreError("Invalid document fragment.");
        }
    }
    exec(db_, "BEGIN IMMEDIATE");
    try {
        exec(db_, "DELETE FROM indexed_files");
        exec(db_, "DELETE FROM index_state");
        exec(db_, "INSERT INTO index_state(id,root) VALUES(1,'')");
        exec(db_, "DELETE FROM embeddings");
        exec(db_, "DELETE FROM embedding_model");
        exec(db_, "DELETE FROM chunks");
        Statement insert(db_, "INSERT INTO chunks(source_path,text,start_byte,end_byte) VALUES(?,?,?,?)");
        Statement insert_embedding(db_, "INSERT INTO embeddings(chunk_id,vector) VALUES(?,?)");
        for (std::size_t position = 0; position < chunks.size(); ++position) {
            const auto& chunk = chunks[position];
            sqlite3_reset(insert.get());
            sqlite3_clear_bindings(insert.get());
            if (sqlite3_bind_text(insert.get(), 1, chunk.source_path.data(), static_cast<int>(chunk.source_path.size()), SQLITE_TRANSIENT) != SQLITE_OK ||
                sqlite3_bind_text(insert.get(), 2, chunk.text.data(), static_cast<int>(chunk.text.size()), SQLITE_TRANSIENT) != SQLITE_OK ||
                sqlite3_bind_int64(insert.get(), 3, static_cast<sqlite3_int64>(chunk.start_byte)) != SQLITE_OK ||
                sqlite3_bind_int64(insert.get(), 4, static_cast<sqlite3_int64>(chunk.end_byte)) != SQLITE_OK ||
                sqlite3_step(insert.get()) != SQLITE_DONE) {
                throw StoreError("Cannot write encrypted index.");
            }
            if (!model_id.empty()) {
                const auto& vector = embeddings[position];
                sqlite3_reset(insert_embedding.get());
                sqlite3_clear_bindings(insert_embedding.get());
                if (sqlite3_bind_int64(insert_embedding.get(), 1, sqlite3_last_insert_rowid(db_)) != SQLITE_OK ||
                    sqlite3_bind_blob(insert_embedding.get(), 2, vector.data(),
                                      static_cast<int>(vector.size() * sizeof(float)), SQLITE_TRANSIENT) != SQLITE_OK ||
                    sqlite3_step(insert_embedding.get()) != SQLITE_DONE) {
                    throw StoreError("Cannot write encrypted embeddings.");
                }
            }
        }
        if (!model_id.empty()) {
            Statement metadata(db_, "INSERT INTO embedding_model(id,model_id) VALUES(1,?)");
            if (sqlite3_bind_text(metadata.get(), 1, model_id.data(), static_cast<int>(model_id.size()), SQLITE_TRANSIENT) != SQLITE_OK ||
                sqlite3_step(metadata.get()) != SQLITE_DONE) {
                throw StoreError("Cannot write encrypted embeddings.");
            }
        }
        exec(db_, "COMMIT");
    } catch (...) {
        sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
        throw;
    }
}

std::string EncryptedStore::read_model_id() const {
    if (version_ == 1) return {};
    Statement statement(db_, "SELECT model_id FROM embedding_model WHERE id=1");
    const int step = sqlite3_step(statement.get());
    if (step == SQLITE_DONE) return {};
    if (step != SQLITE_ROW || sqlite3_column_text(statement.get(), 0) == nullptr) {
        throw StoreError("Encrypted embeddings are damaged.");
    }
    return {reinterpret_cast<const char*>(sqlite3_column_text(statement.get(), 0)),
            static_cast<std::size_t>(sqlite3_column_bytes(statement.get(), 0))};
}

std::pair<std::string, std::map<std::string, std::string>> EncryptedStore::read_file_hashes() const {
    if (version_ < 3) return {};
    Statement state(db_, "SELECT root FROM index_state WHERE id=1");
    const int state_step = sqlite3_step(state.get());
    if (state_step == SQLITE_DONE) {
        throw StoreError("Encrypted index metadata is damaged.");
    }
    if (state_step != SQLITE_ROW || sqlite3_column_text(state.get(), 0) == nullptr) {
        throw StoreError("Encrypted index metadata is damaged.");
    }
    std::string root(reinterpret_cast<const char*>(sqlite3_column_text(state.get(), 0)),
                     static_cast<std::size_t>(sqlite3_column_bytes(state.get(), 0)));
    if (root.empty()) {
        if (scalar_int(db_, "SELECT count(*) FROM indexed_files") != 0) {
            throw StoreError("Encrypted index metadata is damaged.");
        }
        return {};
    }
    Statement rows(db_, "SELECT path,digest FROM indexed_files");
    std::map<std::string, std::string> files;
    int step;
    while ((step = sqlite3_step(rows.get())) == SQLITE_ROW) {
        if (!sqlite3_column_text(rows.get(), 0) || !sqlite3_column_text(rows.get(), 1)) {
            throw StoreError("Encrypted index metadata is damaged.");
        }
        std::string path(reinterpret_cast<const char*>(sqlite3_column_text(rows.get(), 0)),
                         static_cast<std::size_t>(sqlite3_column_bytes(rows.get(), 0)));
        std::string digest(reinterpret_cast<const char*>(sqlite3_column_text(rows.get(), 1)),
                           static_cast<std::size_t>(sqlite3_column_bytes(rows.get(), 1)));
        if (path.empty() || !valid_digest(digest)) throw StoreError("Encrypted index metadata is damaged.");
        files.emplace(std::move(path), std::move(digest));
    }
    if (step != SQLITE_DONE || root.empty() ||
        scalar_int(db_, "SELECT count(*) FROM chunks LEFT JOIN indexed_files ON chunks.source_path=indexed_files.path WHERE indexed_files.path IS NULL") != 0) {
        throw StoreError("Encrypted index metadata is damaged.");
    }
    return {std::move(root), std::move(files)};
}

std::size_t EncryptedStore::apply_update(
    const std::string& root,
    const std::vector<std::pair<std::string, std::string>>& files,
    const std::vector<Chunk>& chunks,
    const std::vector<std::vector<float>>& embeddings,
    const std::string& model_id,
    const std::vector<std::string>& prepared_paths) {
    if (version_ != 3 || root.empty() || root.size() > std::numeric_limits<int>::max() ||
        model_id.size() > std::numeric_limits<int>::max() ||
        (!model_id.empty() && embeddings.size() != chunks.size()) ||
        (model_id.empty() && !embeddings.empty())) {
        throw StoreError("Invalid incremental index update.");
    }
    exec(db_, "BEGIN IMMEDIATE");
    try {
        auto [previous_root, previous_files] = read_file_hashes();
        if (!previous_root.empty() && previous_root != root) {
            throw StoreError("Index belongs to a different folder.");
        }
        const std::string old_model_id = read_model_id();
        if ((!previous_root.empty() || !old_model_id.empty()) && old_model_id != model_id) {
            throw StoreError("Index uses a different embedding model.");
        }
        if (!previous_root.empty()) {
            const int chunk_count = scalar_int(db_, "SELECT count(*) FROM chunks");
            const int vector_count = scalar_int(db_, "SELECT count(*) FROM embeddings");
            const int paired_count = scalar_int(db_, "SELECT count(*) FROM chunks JOIN embeddings ON embeddings.chunk_id=chunks.id");
            if ((!model_id.empty() && (vector_count != chunk_count || paired_count != chunk_count)) ||
                (model_id.empty() && vector_count != 0)) {
                throw StoreError("Encrypted embeddings are damaged.");
            }
        }
        std::map<std::string, std::string> next_files;
        for (const auto& [path, digest] : files) {
            if (path.empty() || path.size() > std::numeric_limits<int>::max() || !valid_digest(digest) ||
                !next_files.emplace(path, digest).second) {
                throw StoreError("Invalid file fingerprint.");
            }
        }
        std::set<std::string> changed;
        for (const auto& [path, digest] : next_files) {
            if (!previous_files.contains(path) || previous_files.at(path) != digest) changed.insert(path);
        }
        if (prepared_paths.size() != changed.size() ||
            std::set<std::string>(prepared_paths.begin(), prepared_paths.end()) != changed) {
            throw StoreError("Index changed during preparation; retry indexing.");
        }
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            const auto& chunk = chunks[i];
            if (!is_valid(chunk) || !changed.contains(chunk.source_path) ||
                chunk.source_path.size() > std::numeric_limits<int>::max() ||
                chunk.text.size() > std::numeric_limits<int>::max() ||
                chunk.start_byte > static_cast<std::uint64_t>(std::numeric_limits<sqlite3_int64>::max()) ||
                chunk.end_byte > static_cast<std::uint64_t>(std::numeric_limits<sqlite3_int64>::max())) {
                throw StoreError("Invalid document fragment.");
            }
            if (!model_id.empty()) {
                const auto& vector = embeddings[i];
                if (vector.size() != embedding_dimensions) throw StoreError("Invalid embedding dimension.");
                double norm = 0;
                for (float value : vector) {
                    if (!std::isfinite(value)) throw StoreError("Invalid embedding value.");
                    norm += static_cast<double>(value) * value;
                }
                if (std::abs(norm - 1.0) > 0.001) throw StoreError("Embedding must be normalized.");
            }
        }
        if (previous_root.empty()) {
            exec(db_, "DELETE FROM embeddings");
            exec(db_, "DELETE FROM embedding_model");
            exec(db_, "DELETE FROM chunks");
        }
        Statement remove_embeddings(db_, "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE source_path=?)");
        Statement remove_chunks(db_, "DELETE FROM chunks WHERE source_path=?");
        Statement remove_file(db_, "DELETE FROM indexed_files WHERE path=?");
        for (const auto& [path, digest] : previous_files) {
            if (next_files.contains(path) && !changed.contains(path)) continue;
            for (sqlite3_stmt* statement : {remove_embeddings.get(), remove_chunks.get(), remove_file.get()}) {
                sqlite3_reset(statement);
                sqlite3_clear_bindings(statement);
                if (sqlite3_bind_text(statement, 1, path.data(), static_cast<int>(path.size()), SQLITE_TRANSIENT) != SQLITE_OK ||
                    sqlite3_step(statement) != SQLITE_DONE) throw StoreError("Cannot update encrypted index.");
            }
        }
        Statement insert_file(db_, "INSERT INTO indexed_files(path,digest) VALUES(?,?)");
        for (const auto& path : changed) {
            const auto& digest = next_files.at(path);
            sqlite3_reset(insert_file.get());
            sqlite3_clear_bindings(insert_file.get());
            if (sqlite3_bind_text(insert_file.get(), 1, path.data(), static_cast<int>(path.size()), SQLITE_TRANSIENT) != SQLITE_OK ||
                sqlite3_bind_text(insert_file.get(), 2, digest.data(), static_cast<int>(digest.size()), SQLITE_TRANSIENT) != SQLITE_OK ||
                sqlite3_step(insert_file.get()) != SQLITE_DONE) throw StoreError("Cannot update encrypted index.");
        }
        Statement insert(db_, "INSERT INTO chunks(source_path,text,start_byte,end_byte) VALUES(?,?,?,?)");
        Statement insert_embedding(db_, "INSERT INTO embeddings(chunk_id,vector) VALUES(?,?)");
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            const auto& chunk = chunks[i];
            sqlite3_reset(insert.get());
            sqlite3_clear_bindings(insert.get());
            if (sqlite3_bind_text(insert.get(), 1, chunk.source_path.data(), static_cast<int>(chunk.source_path.size()), SQLITE_TRANSIENT) != SQLITE_OK ||
                sqlite3_bind_text(insert.get(), 2, chunk.text.data(), static_cast<int>(chunk.text.size()), SQLITE_TRANSIENT) != SQLITE_OK ||
                sqlite3_bind_int64(insert.get(), 3, static_cast<sqlite3_int64>(chunk.start_byte)) != SQLITE_OK ||
                sqlite3_bind_int64(insert.get(), 4, static_cast<sqlite3_int64>(chunk.end_byte)) != SQLITE_OK ||
                sqlite3_step(insert.get()) != SQLITE_DONE) throw StoreError("Cannot update encrypted index.");
            if (!model_id.empty()) {
                sqlite3_reset(insert_embedding.get());
                sqlite3_clear_bindings(insert_embedding.get());
                const auto& vector = embeddings[i];
                if (sqlite3_bind_int64(insert_embedding.get(), 1, sqlite3_last_insert_rowid(db_)) != SQLITE_OK ||
                    sqlite3_bind_blob(insert_embedding.get(), 2, vector.data(), static_cast<int>(vector.size() * sizeof(float)), SQLITE_TRANSIENT) != SQLITE_OK ||
                    sqlite3_step(insert_embedding.get()) != SQLITE_DONE) throw StoreError("Cannot update encrypted embeddings.");
            }
        }
        if (previous_root.empty()) {
            Statement state(db_, "INSERT INTO index_state(id,root) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET root=excluded.root");
            if (sqlite3_bind_text(state.get(), 1, root.data(), static_cast<int>(root.size()), SQLITE_TRANSIENT) != SQLITE_OK ||
                sqlite3_step(state.get()) != SQLITE_DONE) throw StoreError("Cannot update encrypted index.");
            if (!model_id.empty()) {
                exec(db_, "DELETE FROM embedding_model");
                Statement model(db_, "INSERT INTO embedding_model(id,model_id) VALUES(1,?)");
                if (sqlite3_bind_text(model.get(), 1, model_id.data(), static_cast<int>(model_id.size()), SQLITE_TRANSIENT) != SQLITE_OK ||
                    sqlite3_step(model.get()) != SQLITE_DONE) throw StoreError("Cannot update encrypted embeddings.");
            }
        }
        const auto count = static_cast<std::size_t>(scalar_int(db_, "SELECT count(*) FROM chunks"));
        exec(db_, "COMMIT");
        return count;
    } catch (...) {
        sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
        throw;
    }
}

std::pair<std::string, std::vector<std::vector<float>>> EncryptedStore::read_embeddings() const {
    if (version_ == 1) {
        return {};
    }
    Statement metadata(db_, "SELECT model_id FROM embedding_model WHERE id=1");
    const int metadata_step = sqlite3_step(metadata.get());
    if (metadata_step == SQLITE_DONE) {
        return {};
    }
    if (metadata_step != SQLITE_ROW || sqlite3_column_text(metadata.get(), 0) == nullptr) {
        throw StoreError("Encrypted embeddings are damaged.");
    }
    std::string model_id(reinterpret_cast<const char*>(sqlite3_column_text(metadata.get(), 0)),
                         static_cast<std::size_t>(sqlite3_column_bytes(metadata.get(), 0)));
    Statement rows(db_, "SELECT embeddings.vector FROM chunks "
                        "JOIN embeddings ON embeddings.chunk_id=chunks.id ORDER BY chunks.id");
    std::vector<std::vector<float>> vectors;
    int step;
    while ((step = sqlite3_step(rows.get())) == SQLITE_ROW) {
        if (sqlite3_column_type(rows.get(), 0) != SQLITE_BLOB ||
            sqlite3_column_bytes(rows.get(), 0) != static_cast<int>(embedding_dimensions * sizeof(float))) {
            throw StoreError("Encrypted embeddings are damaged.");
        }
        std::vector<float> vector(embedding_dimensions);
        std::memcpy(vector.data(), sqlite3_column_blob(rows.get(), 0), embedding_dimensions * sizeof(float));
        vectors.push_back(std::move(vector));
    }
    if (step != SQLITE_DONE || vectors.size() != static_cast<std::size_t>(scalar_int(db_, "SELECT count(*) FROM chunks"))) {
        throw StoreError("Encrypted embeddings are damaged.");
    }
    return {std::move(model_id), std::move(vectors)};
}

std::vector<Chunk> EncryptedStore::read_chunks() const {
    Statement statement(db_, "SELECT source_path,text,start_byte,end_byte FROM chunks ORDER BY id");
    std::vector<Chunk> chunks;
    int step = SQLITE_ROW;
    while ((step = sqlite3_step(statement.get())) == SQLITE_ROW) {
        const auto* path = sqlite3_column_text(statement.get(), 0);
        const auto* text = sqlite3_column_text(statement.get(), 1);
        const auto start = sqlite3_column_int64(statement.get(), 2);
        const auto end = sqlite3_column_int64(statement.get(), 3);
        if (!path || !text || start < 0 || end < 0) {
            throw StoreError("Encrypted index is damaged.");
        }
        chunks.push_back({std::string(reinterpret_cast<const char*>(path), static_cast<std::size_t>(sqlite3_column_bytes(statement.get(), 0))),
                          std::string(reinterpret_cast<const char*>(text), static_cast<std::size_t>(sqlite3_column_bytes(statement.get(), 1))),
                          static_cast<std::uint64_t>(start), static_cast<std::uint64_t>(end)});
        if (!is_valid(chunks.back())) {
            throw StoreError("Encrypted index is damaged.");
        }
    }
    if (step != SQLITE_DONE) {
        throw StoreError("Encrypted index is damaged.");
    }
    return chunks;
}

} // namespace hsd
