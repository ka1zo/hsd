# Локальная модель для поиска по смыслу

Модель: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), MIT. Закреплённый revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`. HSD использует `onnx/model_O4.onnx` и `onnx/tokenizer.json`; они лежат только в указанном пользователем локальном каталоге и не входят в Git-репозиторий. Первоначальная загрузка — явное отдельное действие.

| Локальное имя | SHA-256 | Размер |
| --- | --- | ---: |
| `model.onnx` | `4654c156f3e4171abc9c716cdb771bf9116455d15ac1aab364aeeede0e3205b0` | 235 052 531 байт |
| `tokenizer.json` | `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39` | 17 082 730 байт |

Для первоначальной установки с доступом к сети загрузите два файла из каталога [`onnx` закреплённого revision](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3/onnx). На macOS/Linux:

```sh
mkdir -p models/e5-small
curl -fL -o models/e5-small/model.onnx 'https://huggingface.co/intfloat/multilingual-e5-small/resolve/614241f622f53c4eeff9890bdc4f31cfecc418b3/onnx/model_O4.onnx'
curl -fL -o models/e5-small/tokenizer.json 'https://huggingface.co/intfloat/multilingual-e5-small/resolve/614241f622f53c4eeff9890bdc4f31cfecc418b3/onnx/tokenizer.json'
python -m pip install '.[ml]'
hsd index /path/to/documents --output /path/to/search.hsd --model-dir models/e5-small
```

На Windows создайте папку `models\e5-small`, скачайте те же два URL в файлы `model.onnx` и `tokenizer.json` и установите пакет `python -m pip install ".[ml]"`. HSD проверяет SHA-256 обоих файлов перед загрузкой. При несовпадении хэша он прекращает работу до создания или изменения индекса. Обычный вызов `index` и Python API `LocalEmbedder` не скачивают файлы и не требуют сети.

Python API `hsd.embeddings.LocalEmbedder(Path("models/e5-small"))` предоставляет `embed_passages(list[str])` и `embed_query(str)`. Для E5 используются префиксы `passage: ` и `query: `, максимум 512 токенов, masked mean pooling и нормализация L2. Оба метода возвращают векторы размерности 384. CPU inference выполняет ONNX Runtime; исполняемый Python-код из репозитория модели не загружается. Эмбеддинги документов записываются вместе с фрагментами в SQLCipher транзакцией. Поиск по ним описан в [руководстве по векторному поиску](vector-search.md).

Локальный прогон macOS ARM64 (Python 3.13.5, ONNX Runtime 1.31.0, tokenizers 0.23.3, модель O4): загрузка модели 0,49 с; среднее время одного короткого запроса 0,003 с за 10 повторов; максимальный RSS процесса около 950 МБ. Это ориентир для данной машины, не гарантия для других документов и ОС. Тест реальной модели запускается с `HSD_TEST_MODEL_DIR=models/e5-small python -m unittest discover -s tests/python -v`; он блокирует сетевые соединения Python на время inference.

Базовые операции поиска по словам не требуют ML-зависимостей. Проверка CI с локальной моделью выполняется на macOS, Ubuntu и Windows; результаты и методика измерений приведены в [отчёте](benchmarks.md).
