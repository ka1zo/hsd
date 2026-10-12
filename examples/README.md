# Пример для первого запуска

В `demo-documents/` находятся три коротких вымышленных TXT- и Markdown-файла на русском и английском. Используйте их, чтобы проверить индексирование и поиск. В них нет личных данных.

Набор документов для измерений хранится отдельно: [`benchmarks/stage8_cases.json`](../benchmarks/stage8_cases.json).

После [установки HSD](../docs/install.md) из корня репозитория выполните:

```sh
hsd index examples/demo-documents --output demo.hsd
hsd search demo.hsd "маяк корабли ночью" --mode lexical
hsd search demo.hsd "осадкомер" --mode lexical
hsd delete-index demo.hsd
```

При индексировании HSD попросит задать пароль, а при поиске — ввести его снова. `delete-index` удалит файл индекса; демонстрационные документы останутся на месте. Для поиска по смыслу сначала установите модель по [инструкции](../docs/install.md), затем передайте `--model-dir models/e5-small` командам `index` и `search`. Выберите режим `--mode vector` или `--mode hybrid`.

В Windows откройте PowerShell и выполните команды из активированного виртуального окружения. Если команда `hsd` недоступна, запустите `.venv\Scripts\hsd.exe`.
