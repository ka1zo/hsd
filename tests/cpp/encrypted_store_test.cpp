#include "hsd/encrypted_store.hpp"
#include <sqlite3.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("hsd-encrypted-store-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".db");
    std::filesystem::remove(path);
    const std::vector<hsd::Chunk> chunks{{"/private/note.txt", "secret paragraph", 0, 16}};
    {
        hsd::EncryptedStore store(path.string(), "test-password", true);
        store.replace_chunks(chunks);
    }
    {
        hsd::EncryptedStore store(path.string(), "test-password", false);
        const auto loaded = store.read_chunks();
        if (loaded.size() != 1 || loaded.front().text != "secret paragraph" ||
            loaded.front().source_path != "/private/note.txt") {
            throw std::runtime_error("encrypted round trip failed");
        }
    }
    std::cerr << "encrypted_store_test: round trip complete\n";
    bool rejected = false;
    try {
        hsd::EncryptedStore store(path.string(), "wrong-password", false);
    } catch (const hsd::StoreError&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("wrong password accepted");
    std::cerr << "encrypted_store_test: wrong password rejected\n";

    sqlite3* raw = nullptr;
    if (sqlite3_open(path.string().c_str(), &raw) != SQLITE_OK ||
        sqlite3_key(raw, "test-password", 13) != SQLITE_OK ||
        sqlite3_exec(raw, "BEGIN IMMEDIATE; UPDATE chunks SET text='changed';", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error("cannot prepare encrypted journal test");
    }
    const auto journal_path = path.string() + "-journal";
    std::ifstream journal(journal_path, std::ios::binary);
    if (!journal) throw std::runtime_error("journal file missing during transaction");
    const std::string journal_bytes(std::istreambuf_iterator<char>{journal}, {});
    if (journal_bytes.find("secret paragraph") != std::string::npos ||
        journal_bytes.find("/private/note.txt") != std::string::npos) {
        throw std::runtime_error("plaintext found in journal");
    }
    journal.close();
    if (sqlite3_exec(raw, "ROLLBACK", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error("cannot roll back journal test");
    }
    sqlite3_close(raw);
    std::cerr << "encrypted_store_test: journal check complete\n";

    if (sqlite3_open(path.string().c_str(), &raw) != SQLITE_OK ||
        sqlite3_key(raw, "test-password", 13) != SQLITE_OK ||
        sqlite3_exec(raw, "PRAGMA user_version=999", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error("cannot prepare future-version test");
    }
    sqlite3_close(raw);
    std::cerr << "encrypted_store_test: future schema written\n";
    rejected = false;
    try {
        hsd::EncryptedStore store(path.string(), "test-password", false);
    } catch (const hsd::StoreError& error) {
        rejected = std::string(error.what()).find("newer") != std::string::npos;
    }
    if (!rejected) throw std::runtime_error("newer schema accepted");
    std::cerr << "encrypted_store_test: newer schema rejected\n";

    if (sqlite3_open(path.string().c_str(), &raw) != SQLITE_OK ||
        sqlite3_key(raw, "test-password", 13) != SQLITE_OK ||
        sqlite3_exec(raw, "PRAGMA user_version=3", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error("cannot restore schema version");
    }
    sqlite3_close(raw);
    std::cerr << "encrypted_store_test: schema restored\n";

    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(100);
        file.put('\0');
    }
    rejected = false;
    try {
        hsd::EncryptedStore store(path.string(), "test-password", false);
    } catch (const hsd::StoreError&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("corrupt database accepted");
    std::cerr << "encrypted_store_test: corrupt database rejected\n";
    std::filesystem::remove(path);

    const auto legacy = path.string() + ".legacy";
    if (sqlite3_open(legacy.c_str(), &raw) != SQLITE_OK ||
        sqlite3_key(raw, "test-password", 13) != SQLITE_OK ||
        sqlite3_exec(raw, "CREATE TABLE chunks (id INTEGER PRIMARY KEY, source_path TEXT NOT NULL, text TEXT NOT NULL, start_byte INTEGER NOT NULL, end_byte INTEGER NOT NULL);"
                          "INSERT INTO chunks(source_path,text,start_byte,end_byte) VALUES('/note.txt','legacy text',0,11);"
                          "PRAGMA user_version=1", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error("cannot prepare legacy index");
    }
    sqlite3_close(raw);
    {
        hsd::EncryptedStore old_store(legacy, "test-password", false);
        if (old_store.read_chunks().front().text != "legacy text" ||
            !old_store.read_embeddings().second.empty()) {
            throw std::runtime_error("legacy index could not be read");
        }
    }
    {
        hsd::EncryptedStore upgraded(legacy, "test-password", true);
        if (upgraded.read_chunks().front().text != "legacy text" ||
            !upgraded.read_embeddings().second.empty()) {
            throw std::runtime_error("legacy migration lost chunks");
        }
    }
    std::filesystem::remove(legacy);

    const auto version_two = path.string() + ".v2";
    if (sqlite3_open(version_two.c_str(), &raw) != SQLITE_OK ||
        sqlite3_key(raw, "test-password", 13) != SQLITE_OK ||
        sqlite3_exec(raw, "CREATE TABLE chunks (id INTEGER PRIMARY KEY, source_path TEXT NOT NULL, text TEXT NOT NULL, start_byte INTEGER NOT NULL, end_byte INTEGER NOT NULL);"
                          "CREATE TABLE embedding_model (id INTEGER PRIMARY KEY CHECK(id=1), model_id TEXT NOT NULL);"
                          "CREATE TABLE embeddings (chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id), vector BLOB NOT NULL);"
                          "INSERT INTO chunks(source_path,text,start_byte,end_byte) VALUES('/documents/old.txt','old',0,3);"
                          "PRAGMA user_version=2", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error("cannot prepare version 2 index");
    }
    sqlite3_close(raw);
    {
        hsd::EncryptedStore readable(version_two, "test-password", false);
        if (readable.read_chunks().size() != 1 || !readable.read_file_hashes().second.empty()) {
            throw std::runtime_error("version 2 index could not be read");
        }
    }
    {
        hsd::EncryptedStore upgraded(version_two, "test-password", true);
        if (upgraded.apply_update("/documents", {{"/documents/new.txt", std::string(64, 'c')}},
                                  {{"/documents/new.txt", "new", 0, 3}}, {}, "",
                                  {"/documents/new.txt"}) != 1 ||
            upgraded.read_chunks().front().text != "new") {
            throw std::runtime_error("version 2 migration failed");
        }
    }
    std::filesystem::remove(version_two);

    const auto incremental = path.string() + ".incremental";
    const std::string first_digest(64, 'a');
    const std::string second_digest(64, 'b');
    {
        hsd::EncryptedStore store(incremental, "test-password", true);
        if (store.apply_update("/documents", {{"/documents/keep.txt", first_digest},
                                               {"/documents/change.txt", first_digest}},
                               {{"/documents/keep.txt", "steady", 0, 6},
                                {"/documents/change.txt", "old", 0, 3}}, {}, "",
                               {"/documents/keep.txt", "/documents/change.txt"}) != 2) {
            throw std::runtime_error("initial incremental update failed");
        }
    }
    auto chunk_id = [&](const char* source) {
        sqlite3* connection = nullptr;
        if (sqlite3_open(incremental.c_str(), &connection) != SQLITE_OK ||
            sqlite3_key(connection, "test-password", 13) != SQLITE_OK) {
            throw std::runtime_error("cannot read encrypted row id");
        }
        sqlite3_stmt* query = nullptr;
        if (sqlite3_prepare_v2(connection, "SELECT id FROM chunks WHERE source_path=?", -1, &query, nullptr) != SQLITE_OK ||
            sqlite3_bind_text(query, 1, source, -1, SQLITE_TRANSIENT) != SQLITE_OK ||
            sqlite3_step(query) != SQLITE_ROW) {
            throw std::runtime_error("cannot read encrypted row id");
        }
        const auto id = sqlite3_column_int64(query, 0);
        sqlite3_finalize(query);
        sqlite3_close(connection);
        return id;
    };
    const auto keep_id = chunk_id("/documents/keep.txt");
    {
        hsd::EncryptedStore store(incremental, "test-password", true);
        if (store.apply_update("/documents", {{"/documents/keep.txt", first_digest},
                                               {"/documents/change.txt", second_digest}},
                               {{"/documents/change.txt", "new", 0, 3}}, {}, "",
                               {"/documents/change.txt"}) != 2) {
            throw std::runtime_error("incremental update count is wrong");
        }
    }
    if (chunk_id("/documents/keep.txt") != keep_id) {
        throw std::runtime_error("unchanged fragment was rewritten");
    }
    rejected = false;
    {
        hsd::EncryptedStore store(incremental, "test-password", true);
        try {
            (void)store.apply_update("/documents", {{"/documents/keep.txt", first_digest},
                                                     {"/documents/change.txt", second_digest}},
                                     {}, {}, "", {"/documents/keep.txt"});
        } catch (const hsd::StoreError&) {
            rejected = true;
        }
        if (!rejected || store.read_chunks().size() != 2) {
            throw std::runtime_error("stale preparation was accepted");
        }
    }
    if (sqlite3_open(incremental.c_str(), &raw) != SQLITE_OK ||
        sqlite3_key(raw, "test-password", 13) != SQLITE_OK ||
        sqlite3_exec(raw, "CREATE TRIGGER reject_fragment BEFORE INSERT ON chunks "
                          "WHEN NEW.text='fail' BEGIN SELECT RAISE(ABORT,'injected failure'); END", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error("cannot prepare rollback test");
    }
    sqlite3_close(raw);
    rejected = false;
    {
        hsd::EncryptedStore store(incremental, "test-password", true);
        try {
            (void)store.apply_update("/documents", {{"/documents/keep.txt", second_digest}},
                               {{"/documents/keep.txt", "fail", 0, 4}}, {}, "",
                               {"/documents/keep.txt"});
        } catch (const hsd::StoreError&) {
            rejected = true;
        }
        if (!rejected || store.read_chunks().size() != 2 ||
            store.read_file_hashes().second.at("/documents/keep.txt") != first_digest) {
            throw std::runtime_error("failed update did not roll back");
        }
    }
    if (sqlite3_open(incremental.c_str(), &raw) != SQLITE_OK ||
        sqlite3_key(raw, "test-password", 13) != SQLITE_OK ||
        sqlite3_exec(raw, "UPDATE indexed_files SET digest='damaged' WHERE path='/documents/keep.txt'", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error("cannot prepare metadata damage test");
    }
    sqlite3_close(raw);
    rejected = false;
    try {
        hsd::EncryptedStore store(incremental, "test-password", false);
        (void)store.read_file_hashes();
    } catch (const hsd::StoreError&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("damaged file metadata accepted");
    if (sqlite3_open(incremental.c_str(), &raw) != SQLITE_OK ||
        sqlite3_key(raw, "test-password", 13) != SQLITE_OK ||
        sqlite3_exec(raw, "UPDATE indexed_files SET digest='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' "
                          "WHERE path='/documents/keep.txt'; DELETE FROM index_state; DELETE FROM indexed_files", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error("cannot prepare missing state test");
    }
    sqlite3_close(raw);
    rejected = false;
    {
        hsd::EncryptedStore store(incremental, "test-password", true);
        try {
            (void)store.read_file_hashes();
            (void)store.apply_update("/documents", {{"/documents/keep.txt", second_digest}},
                                     {{"/documents/keep.txt", "replacement", 0, 11}}, {}, "",
                                     {"/documents/keep.txt"});
        } catch (const hsd::StoreError&) {
            rejected = true;
        }
        if (!rejected || store.read_chunks().size() != 2) {
            throw std::runtime_error("missing state silently replaced index");
        }
    }
    std::filesystem::remove(incremental);

    const auto empty_state = path.string() + ".empty-state";
    {
        hsd::EncryptedStore store(empty_state, "test-password", true);
        if (store.apply_update("/empty", {}, {}, {}, "", {}) != 0) {
            throw std::runtime_error("cannot prepare empty index test");
        }
    }
    if (sqlite3_open(empty_state.c_str(), &raw) != SQLITE_OK ||
        sqlite3_key(raw, "test-password", 13) != SQLITE_OK ||
        sqlite3_exec(raw, "DELETE FROM index_state", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error("cannot remove empty index state");
    }
    sqlite3_close(raw);
    rejected = false;
    try {
        hsd::EncryptedStore store(empty_state, "test-password", false);
        (void)store.read_file_hashes();
    } catch (const hsd::StoreError&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("empty index missing state accepted");
    std::filesystem::remove(empty_state);

    const auto unaligned = path.string() + ".unaligned";
    {
        hsd::EncryptedStore store(unaligned, "test-password", true);
        std::vector<float> first(384, 0.0f);
        std::vector<float> second(384, 0.0f);
        first[0] = 1.0f;
        second[1] = 1.0f;
        store.replace_chunks_with_embeddings(
            {{"/a.txt", "first", 0, 5}, {"/b.txt", "second", 0, 6}},
            {first, second}, "test-model");
    }
    if (sqlite3_open(unaligned.c_str(), &raw) != SQLITE_OK ||
        sqlite3_key(raw, "test-password", 13) != SQLITE_OK ||
        sqlite3_exec(raw, "INSERT INTO embeddings(chunk_id,vector) SELECT 999,vector FROM embeddings WHERE chunk_id=1;"
                          "DELETE FROM embeddings WHERE chunk_id=1", nullptr, nullptr, nullptr) != SQLITE_OK) {
        throw std::runtime_error("cannot prepare unaligned embeddings test");
    }
    sqlite3_close(raw);
    rejected = false;
    try {
        hsd::EncryptedStore store(unaligned, "test-password", false);
        (void)store.read_embeddings();
    } catch (const hsd::StoreError&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("unaligned embeddings accepted");
    std::filesystem::remove(unaligned);
}
