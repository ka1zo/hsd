"""Local Qt Widgets interface for HSD's encrypted document search."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QMetaObject, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from . import _core
from .application import SearchHit, SearchSession, create_or_update, delete_index, open_session
from .embeddings import ModelError
from .ingest import DocumentReadError


def _safe_error(error: Exception) -> str:
    if isinstance(error, _core.StoreError):
        return str(error)
    if isinstance(error, ModelError):
        return str(error)
    if isinstance(error, DocumentReadError):
        return "Cannot read a document in the selected folder. Check its UTF-8 encoding and size."
    if isinstance(error, ValueError):
        return str(error)
    if isinstance(error, OSError):
        return "File operation failed. Check access to the selected folder and index."
    return "Operation failed. The index was not changed by the interface."


class _Worker(QObject):
    done = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.session: SearchSession | None = None

    @Slot(object)
    def run(self, request: tuple[str, tuple]) -> None:
        action, args = request
        try:
            if action == "create":
                root, index, password, model_dir = args
                count = create_or_update(root, index, password, model_dir)
                self._close()
                self.session = open_session(index, password, model_dir)
                value = (count, self.session.has_model, self.session.root)
            elif action == "open":
                index, password, model_dir = args
                self._close()
                self.session = open_session(index, password, model_dir)
                value = (self.session.size, self.session.has_model, self.session.root)
            elif action == "search":
                if self.session is None:
                    raise ValueError("open an encrypted index first")
                value = self.session.search(*args)
            elif action == "lock":
                self._close()
                value = None
            elif action == "delete":
                self._close()
                delete_index(args[0])
                value = None
            else:
                raise ValueError("unsupported operation")
            self.done.emit(action, value)
        except Exception as error:
            if action in ("create", "open", "delete"):
                self._close()
            self.failed.emit(action, _safe_error(error))

    @Slot()
    def stop(self) -> None:
        self._close()

    def _close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None


class DesktopWindow(QMainWindow):
    request = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HSD — Hybrid Search for Documents")
        self.resize(820, 600)
        self.busy = False
        self._closed = False
        self._ready = False
        self._has_model = False
        self._root = ""
        self._hits: list[SearchHit] = []
        self._search_mode = "lexical"

        self.folder_edit = QLineEdit()
        self.index_edit = QLineEdit()
        self.model_edit = QLineEdit()
        form = QFormLayout()
        form.addRow("Document folder", self._path_row(self.folder_edit, self._choose_folder))
        form.addRow("Encrypted index", self._index_row())
        form.addRow("Local model (optional)", self._path_row(self.model_edit, self._choose_model))

        self.create_button = QPushButton("Create / update index")
        self.open_button = QPushButton("Open index")
        self.lock_button = QPushButton("Lock")
        self.delete_button = QPushButton("Delete index")
        for button, handler in (
            (self.create_button, self._create_clicked),
            (self.open_button, self._open_clicked),
            (self.lock_button, self.lock_index),
            (self.delete_button, self._delete_clicked),
        ):
            button.clicked.connect(handler)
        actions = QHBoxLayout()
        for button in (self.create_button, self.open_button, self.lock_button,
                       self.delete_button):
            actions.addWidget(button)

        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Search documents")
        self.query_edit.returnPressed.connect(self._search_clicked)
        self.mode_combo = QComboBox()
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self._search_clicked)
        search_row = QHBoxLayout()
        search_row.addWidget(self.query_edit, 1)
        search_row.addWidget(self.mode_combo)
        search_row.addWidget(self.search_button)

        self.results = QListWidget()
        self.open_source_button = QPushButton("Open selected source")
        self.open_source_button.clicked.connect(self._open_source)
        self.status_label = QLabel("Choose a folder or open an encrypted index.")
        self.status_label.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(actions)
        layout.addLayout(search_row)
        layout.addWidget(self.results, 1)
        layout.addWidget(self.open_source_button)
        layout.addWidget(self.status_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self._thread = QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self.request.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()
        self._refresh_controls()

    def _path_row(self, edit: QLineEdit, browse) -> QWidget:
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(browse)
        row.addWidget(button)
        wrapper = QWidget()
        wrapper.setLayout(row)
        return wrapper

    def _index_row(self) -> QWidget:
        row = QHBoxLayout()
        row.addWidget(self.index_edit, 1)
        new_button = QPushButton("New…")
        new_button.clicked.connect(self._choose_new_index)
        row.addWidget(new_button)
        open_button = QPushButton("Browse…")
        open_button.clicked.connect(self._choose_index)
        row.addWidget(open_button)
        wrapper = QWidget()
        wrapper.setLayout(row)
        return wrapper

    def _choose_new_index(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Choose new encrypted index", "", "HSD index (*.hsd)")
        if path:
            self.index_edit.setText(path if path.endswith(".hsd") else path + ".hsd")

    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose document folder")
        if path:
            self.folder_edit.setText(path)

    def _choose_index(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose encrypted index", "", "HSD index (*.hsd)")
        if path:
            self.index_edit.setText(path)

    def _choose_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose local E5 model")
        if path:
            self.model_edit.setText(path)

    def _password(self, confirm: bool = False) -> str | None:
        value, accepted = QInputDialog.getText(self, "Unlock HSD", "Index password:",
                                               QLineEdit.EchoMode.Password)
        if not accepted:
            return None
        if not value:
            self.status_label.setText("Index password cannot be empty.")
            return None
        if confirm:
            second, accepted = QInputDialog.getText(self, "Confirm password",
                                                    "Repeat index password:",
                                                    QLineEdit.EchoMode.Password)
            if not accepted or value != second:
                self.status_label.setText("Passwords do not match.")
                return None
        return value

    def _create_clicked(self) -> None:
        folder = self.folder_edit.text().strip()
        index = self.index_edit.text().strip()
        if not folder or not index:
            self.status_label.setText("Choose a document folder and a .hsd index path.")
            return
        password = self._password(confirm=not Path(index).exists())
        if password is not None:
            self.create_index(Path(folder), Path(index), password,
                              Path(self.model_edit.text()) if self.model_edit.text() else None)

    def _open_clicked(self) -> None:
        index = self.index_edit.text().strip()
        if not index:
            self.status_label.setText("Choose a .hsd index first.")
            return
        password = self._password()
        if password is not None:
            self.open_index(Path(index), password,
                            Path(self.model_edit.text()) if self.model_edit.text() else None)

    def _delete_clicked(self) -> None:
        index = self.index_edit.text().strip()
        if not index:
            self.status_label.setText("Choose a .hsd index first.")
            return
        answer = QMessageBox.question(self, "Delete encrypted index",
                                      f"Delete this index?\n{index}\nSource documents stay untouched.")
        if answer == QMessageBox.StandardButton.Yes:
            self.delete_index(Path(index))

    def _search_clicked(self) -> None:
        self.search(self.query_edit.text(), self.mode_combo.currentData())

    def _start(self, action: str, *args) -> None:
        if self.busy:
            return
        if action in ("create", "open"):
            self._ready = False
            self._has_model = False
            self._root = ""
            self._hits = []
            self.results.clear()
            self.query_edit.clear()
        self.busy = True
        self.status_label.setText(f"{action.capitalize()} in progress…")
        self._refresh_controls()
        self.request.emit((action, args))

    def create_index(self, root: Path, index_path: Path, password: str,
                     model_dir: Path | None = None) -> None:
        self.folder_edit.setText(str(root))
        self.index_edit.setText(str(index_path))
        self._start("create", root, index_path, password, model_dir)

    def open_index(self, index_path: Path, password: str,
                   model_dir: Path | None = None) -> None:
        self.index_edit.setText(str(index_path))
        self._start("open", index_path, password, model_dir)

    def search(self, query: str, mode: str = "lexical") -> None:
        if not self._ready:
            self.status_label.setText("Open an encrypted index first.")
            return
        if self.busy:
            return
        self._search_mode = mode
        self._start("search", query, mode, 10)

    def lock_index(self) -> None:
        self._ready = False
        self._hits = []
        self.results.clear()
        self.query_edit.clear()
        self._refresh_controls()
        self._start("lock")

    def delete_index(self, index_path: Path) -> None:
        self._ready = False
        self._hits = []
        self.results.clear()
        self.query_edit.clear()
        self._refresh_controls()
        self._start("delete", index_path)

    @Slot(str, object)
    def _on_done(self, action: str, value: object) -> None:
        self.busy = False
        if action in ("create", "open"):
            count, self._has_model, self._root = value
            self._ready = True
            self.status_label.setText(f"Encrypted index open: {count} fragments.")
            if action == "create":
                self.status_label.setText(f"Index updated and open: {count} fragments.")
            if not self._root:
                self.status_label.setText(
                    "Legacy index open. Rebuild it to enable opening source files.")
        elif action == "search":
            self._hits = value
            self.results.clear()
            for hit in self._hits:
                if self._search_mode == "hybrid":
                    scores = (f"bm25={hit.lexical_score:.4f} "
                              f"cosine={hit.vector_score:.4f} rrf={hit.hybrid_score:.4f}")
                elif self._search_mode == "vector":
                    scores = f"score={hit.vector_score:.4f}"
                else:
                    scores = f"score={hit.lexical_score:.4f}"
                self.results.addItem(
                    f"{hit.source_path}:{hit.start_byte}-{hit.end_byte} {scores}\n{hit.text}")
            self.status_label.setText(f"{len(self._hits)} matching fragments.")
        else:
            self._ready = False
            self._has_model = False
            self._root = ""
            self.status_label.setText("Index deleted." if action == "delete" else "Index locked.")
        self._refresh_controls()

    @Slot(str, str)
    def _on_failed(self, action: str, message: str) -> None:
        self.busy = False
        if action in ("open", "create", "delete"):
            self._ready = False
            self._has_model = False
            self._hits = []
            self.results.clear()
        self.status_label.setText(message)
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        self.create_button.setEnabled(not self.busy)
        self.open_button.setEnabled(not self.busy)
        self.delete_button.setEnabled(not self.busy)
        self.lock_button.setEnabled(self._ready and not self.busy)
        self.search_button.setEnabled(self._ready and not self.busy)
        self.query_edit.setEnabled(self._ready and not self.busy)
        self.open_source_button.setEnabled(
            self._ready and bool(self._hits) and bool(self._root) and not self.busy)
        self.open_source_button.setToolTip(
            "Rebuild this legacy index to enable opening source files." if self._ready and
            not self._root else "")
        current = self.mode_combo.currentData()
        desired = ("lexical", "vector", "hybrid") if self._has_model else ("lexical",)
        if tuple(self.mode_combo.itemData(i) for i in range(self.mode_combo.count())) != desired:
            self.mode_combo.clear()
            for label, mode in (("Words", "lexical"), ("Meaning", "vector"),
                                ("Mixed", "hybrid")):
                if mode in desired:
                    self.mode_combo.addItem(label, mode)
            previous = self.mode_combo.findData(current)
            if previous >= 0:
                self.mode_combo.setCurrentIndex(previous)

    def _open_source(self) -> None:
        if not self._root:
            self.status_label.setText(
                "Rebuild this legacy index to enable opening source files.")
            return
        position = self.results.currentRow()
        if position < 0 or position >= len(self._hits):
            return
        try:
            root = Path(self._root).resolve(strict=True)
            source = Path(self._hits[position].source_path).resolve(strict=True)
            if not source.is_file() or not source.is_relative_to(root):
                raise ValueError("Source file is unavailable inside the indexed folder.")
        except (OSError, ValueError) as error:
            self.status_label.setText(_safe_error(error))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(source)))

    def closeEvent(self, event) -> None:
        if self._closed:
            event.accept()
            return
        if self.busy:
            self.status_label.setText("Wait for the current operation to finish before closing.")
            event.ignore()
            return
        QMetaObject.invokeMethod(self._worker, "stop", Qt.ConnectionType.BlockingQueuedConnection)
        self._thread.quit()
        self._thread.wait()
        self._closed = True
        self._ready = False
        self._has_model = False
        self._root = ""
        self._hits = []
        self.results.clear()
        self.folder_edit.clear()
        self.index_edit.clear()
        self.model_edit.clear()
        self.query_edit.clear()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = DesktopWindow()
    window.show()
    return app.exec()
