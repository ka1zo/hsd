"""Explicitly fetch the pinned public E5 model and verify both files."""

import argparse
import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen


REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
BASE_URL = f"https://huggingface.co/intfloat/multilingual-e5-small/resolve/{REVISION}/onnx"
FILES = {
    "model.onnx": ("model_O4.onnx", "4654c156f3e4171abc9c716cdb771bf9116455d15ac1aab364aeeede0e3205b0"),
    "tokenizer.json": ("tokenizer.json", "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(url: str, destination: Path, expected_sha256: str) -> None:
    """Replace a file only after its complete download matches the pinned hash."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".partial",
                                         delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            with urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {destination.name}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_model(destination: Path) -> None:
    for name, (remote_name, digest) in FILES.items():
        download_one(f"{BASE_URL}/{remote_name}", destination / name, digest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("models/e5-small"))
    options = parser.parse_args()
    fetch_model(options.dest)
    print(f"Verified pinned E5 model in {options.dest}")


if __name__ == "__main__":
    main()
