# Установка и первый запуск

HSD пока устанавливается из исходного кода. Готовых установщиков и подписанных приложений нет. Проверенная конфигурация требует Python 3.11–3.13, компилятор C++20, CMake 3.20+ и SQLCipher Community Edition 4.x.

Скачайте ZIP-архив через кнопку **Code → Download ZIP** на GitHub или клонируйте репозиторий командой `git clone https://github.com/ka1zo/hsd.git`. Дальнейшие команды выполняйте в папке HSD.

Поиск по словам не требует модели. Для поиска по смыслу установите дополнительные Python-пакеты и отдельно загрузите модель размером около 252 МБ. Индексация и поиск выполняются на компьютере пользователя. Интернет нужен для установки зависимостей и первоначальной загрузки модели.

## macOS

Установите инструменты Xcode Command Line Tools, Homebrew и зависимости:

```sh
xcode-select --install
brew install cmake pkg-config sqlcipher python@3.12
```

Из корня клонированного HSD создайте окружение и установите пакет:

```sh
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
hsd --help
```

## Linux (Ubuntu 24.04+)

Установите системные заголовки SQLCipher, OpenSSL и инструменты сборки:

```sh
sudo apt update
sudo apt install -y build-essential cmake pkg-config libsqlcipher-dev libssl-dev libegl1 python3-venv python3-pip
```

Затем создайте окружение и установите пакет из корня HSD:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
hsd --help
```

Если пакет дистрибутива не устанавливает файл `sqlcipher.pc`, соберите SQLCipher Community Edition по [инструкции для Linux](https://www.zetetic.net/sqlcipher/sqlcipher-linux-community/) и добавьте каталог `pkgconfig` в `PKG_CONFIG_PATH`.

## Windows 10/11 x64

Установите Visual Studio 2022 Build Tools с рабочей нагрузкой **Desktop development with C++**, CMake и Git. Затем установите vcpkg по [инструкции Microsoft](https://learn.microsoft.com/vcpkg/get_started/get-started) и укажите путь к нему в переменной `VCPKG_ROOT`. В PowerShell, находясь в корне HSD, выполните:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
& "$env:VCPKG_ROOT\vcpkg.exe" install sqlcipher:x64-windows-static-md
$toolchain = (Join-Path $env:VCPKG_ROOT 'scripts/buildsystems/vcpkg.cmake').Replace('\', '/')
$env:CMAKE_ARGS = "-DCMAKE_TOOLCHAIN_FILE=$toolchain -DVCPKG_TARGET_TRIPLET=x64-windows-static-md"
python -m pip install .
hsd --help
```

Если PowerShell запрещает запуск скрипта активации окружения, используйте `.\.venv\Scripts\python.exe -m pip ...` и `.\.venv\Scripts\hsd.exe` напрямую. Для Python-сборки C++ расширения переменная `CMAKE_ARGS` должна содержать путь к toolchain vcpkg.

## Индексирование и поиск по словам

Создайте зашифрованный индекс. Программа запросит пароль интерактивно; задавать его аргументом командной строки не нужно.

```sh
hsd index examples/demo-documents --output demo.hsd
hsd search demo.hsd "маяк корабли ночью" --mode lexical
```

Пароль нельзя восстановить. Если он потерян, индекс удаляется и создаётся заново из исходных документов. Удаление индекса не затрагивает исходные файлы.

## Поиск по смыслу

Для поиска по смыслу установите дополнительные пакеты и загрузите файлы модели. Команда проверяет SHA-256 файлов и сохраняет их только после успешной загрузки:

```sh
python -m pip install '.[ml]'
python -m scripts.fetch_model --dest models/e5-small
hsd index examples/demo-documents --output demo.hsd --model-dir models/e5-small
hsd search demo.hsd "суда ночью у берега" --mode vector --model-dir models/e5-small
hsd search demo.hsd "суда ночью у берега" --mode hybrid --model-dir models/e5-small
```

На Windows та же команда использует установленные файлы `.venv\Scripts\hsd.exe` и Python из активного окружения. Лицензия и точные хэши файлов модели описаны в [описании модели](model.md). Модель не загружается при запуске HSD.

## Настольное окно

В том же окружении установите Qt Widgets и, если нужен поиск по смыслу, ML-зависимости:

```sh
python -m pip install '.[desktop,ml]'
hsd-desktop
```

Для одного лишь поиска по словам достаточно `python -m pip install '.[desktop]'`; модель можно не загружать. На Windows запускайте `.\.venv\Scripts\hsd-desktop.exe` из PowerShell. Окно предлагает выбрать папку, создать новый `.hsd` через кнопку **New…**, ввести пароль дважды и искать. Для повторного открытия выберите существующий индекс кнопкой **Browse…** и введите пароль. Указанная модель должна совпадать с моделью, использованной при индексации; без модели доступны только слова. Кнопка удаления затрагивает индекс, но не исходные документы.

Окно работает, пока запущена программа и открыт индекс. Пароль и историю запросов оно не сохраняет. Закройте окно или нажмите **Lock**, чтобы завершить поисковую сессию. Если пароль потерян, удалите индекс и создайте его заново из исходной папки. Для первой проверки подойдут файлы из `examples/demo-documents`; индекс можно сохранить как `demo.hsd`.

## Проверки после установки

Из корня репозитория:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
python -m unittest discover -s tests/python -v
```

На Windows добавьте `--config Release` к `cmake --build` и `ctest`. Для полного набора тестов с проверкой локальной модели:

```sh
python -m scripts.fetch_model --dest models/e5-small
HSD_TEST_MODEL_DIR=models/e5-small python -m unittest discover -s tests/python -v
python -m benchmarks.stage8_eval --model-dir models/e5-small --k 3 --repeats 10
```

В PowerShell сначала задайте переменную `$env:HSD_TEST_MODEL_DIR = 'models/e5-small'`. Сборка и основные проверки также запускаются в [GitHub Actions для macOS, Linux и Windows](../.github/workflows/ci.yml). Результаты измерений приведены в [отчёте о производительности](benchmarks.md).

Подробный обзор компонентов и работа в VS Code описаны в [руководстве разработчика](development.md). Ограничения шифрования и восстановление индекса — в [описании формата индекса](index-format.md).
