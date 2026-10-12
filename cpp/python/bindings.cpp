#include "hsd/lexical_index.hpp"
#include "hsd/vector_index.hpp"
#include "hsd/hybrid_index.hpp"
#include "hsd/encrypted_store.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <utility>

namespace py = pybind11;

PYBIND11_MODULE(_core, module) {
    module.doc() = "HSD in-memory C++ lexical, vector and hybrid search core";

    py::class_<hsd::Chunk>(module, "Chunk")
        .def(py::init([](std::string source_path, std::string text,
                         std::uint64_t start_byte, std::uint64_t end_byte) {
                 return hsd::Chunk{std::move(source_path), std::move(text),
                                   start_byte, end_byte};
             }),
             py::arg("source_path"), py::arg("text"),
             py::arg("start_byte"), py::arg("end_byte"))
        .def_readonly("source_path", &hsd::Chunk::source_path)
        .def_readonly("text", &hsd::Chunk::text)
        .def_readonly("start_byte", &hsd::Chunk::start_byte)
        .def_readonly("end_byte", &hsd::Chunk::end_byte);

    py::class_<hsd::SearchResult>(module, "SearchResult")
        .def_readonly("chunk", &hsd::SearchResult::chunk)
        .def_readonly("lexical_score", &hsd::SearchResult::lexical_score);

    py::class_<hsd::VectorSearchResult>(module, "VectorSearchResult")
        .def_readonly("chunk", &hsd::VectorSearchResult::chunk)
        .def_readonly("vector_score", &hsd::VectorSearchResult::vector_score);

    py::class_<hsd::VectorIndex>(module, "VectorIndex")
        .def(py::init<>())
        .def("add_chunks", &hsd::VectorIndex::add_chunks,
             py::arg("chunks"), py::arg("embeddings"))
        .def("search", &hsd::VectorIndex::search, py::arg("query"),
             py::arg("limit") = 10)
        .def_property_readonly("size", &hsd::VectorIndex::size);

    py::class_<hsd::HybridSearchResult>(module, "HybridSearchResult")
        .def_readonly("chunk", &hsd::HybridSearchResult::chunk)
        .def_readonly("lexical_score", &hsd::HybridSearchResult::lexical_score)
        .def_readonly("vector_score", &hsd::HybridSearchResult::vector_score)
        .def_readonly("hybrid_score", &hsd::HybridSearchResult::hybrid_score);

    py::class_<hsd::HybridIndex>(module, "HybridIndex")
        .def(py::init<>())
        .def("add_chunks", &hsd::HybridIndex::add_chunks,
             py::arg("chunks"), py::arg("embeddings"))
        .def("search", &hsd::HybridIndex::search,
             py::arg("query"), py::arg("query_embedding"), py::arg("limit") = 10)
        .def_property_readonly("size", &hsd::HybridIndex::size);

    py::class_<hsd::LexicalIndex>(module, "LexicalIndex")
        .def(py::init<>())
        .def("add_chunks", &hsd::LexicalIndex::add_chunks, py::arg("chunks"))
        .def("search", &hsd::LexicalIndex::search, py::arg("query"),
             py::arg("limit") = 10)
        .def_property_readonly("size", &hsd::LexicalIndex::size);

    py::register_exception<hsd::StoreError>(module, "StoreError");
    py::class_<hsd::EncryptedStore>(module, "EncryptedStore")
        .def(py::init<const std::string&, const std::string&, bool>(),
             py::arg("path"), py::arg("password"), py::arg("create"))
        .def("replace_chunks", &hsd::EncryptedStore::replace_chunks)
        .def("replace_chunks_with_embeddings", &hsd::EncryptedStore::replace_chunks_with_embeddings,
             py::arg("chunks"), py::arg("embeddings"), py::arg("model_id"))
        .def("read_chunks", &hsd::EncryptedStore::read_chunks)
        .def("read_embeddings", &hsd::EncryptedStore::read_embeddings)
        .def("read_model_id", &hsd::EncryptedStore::read_model_id)
        .def("read_file_hashes", &hsd::EncryptedStore::read_file_hashes)
        .def("apply_update", &hsd::EncryptedStore::apply_update,
             py::arg("root"), py::arg("files"), py::arg("chunks"),
             py::arg("embeddings"), py::arg("model_id"), py::arg("prepared_paths"));
}
