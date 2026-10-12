# Third-party components

HSD's own source is licensed under [MIT](LICENSE). The dependencies below
retain their own licenses. Source installations fetch these packages
separately. Any future bundled binary release must include the exact license
texts for the versions and components actually shipped.

| Component | Use | License and upstream notice |
| --- | --- | --- |
| [SQLCipher Community Edition](https://github.com/sqlcipher/sqlcipher) | Encrypted `.hsd` storage | [BSD-style](https://github.com/sqlcipher/sqlcipher/blob/master/LICENSE.md) |
| [OpenSSL](https://www.openssl.org/) | SQLCipher crypto provider | [Apache-2.0](https://github.com/openssl/openssl/blob/master/LICENSE.txt) |
| [pybind11](https://github.com/pybind/pybind11) | C++/Python binding | [BSD-3-Clause](https://github.com/pybind/pybind11/blob/master/LICENSE) |
| [scikit-build-core](https://github.com/scikit-build/scikit-build-core) | Build backend | [Apache-2.0](https://github.com/scikit-build/scikit-build-core/blob/main/LICENSE) |
| [NumPy](https://numpy.org/) | Local ML arrays | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| [ONNX Runtime](https://onnxruntime.ai/) | Offline inference | [MIT](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) |
| [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) | Local tokenization | [Apache-2.0](https://github.com/huggingface/tokenizers/blob/main/LICENSE) |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | Optional local embedding model | [MIT](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/LICENSE) |
| [Qt for Python / PySide6-Essentials](https://doc.qt.io/qtforpython-6/) | Optional desktop window | [LGPLv3/GPLv3 or commercial](https://doc.qt.io/qtforpython-6/licenses.html); deployment obligations depend on the chosen distribution |

The model is downloaded only by an explicit setup command and checked against
the pinned hashes in the [model guide](docs/model.md). It is not part of
the HSD source repository.
