import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("HELIOBEAMAI_ARTIFACTS", REPO / "artifacts"))


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"missing {path} (run `heliobeamai download`)")
    return path


@pytest.fixture
def artifact():
    return lambda rel: _require(ARTIFACTS / rel)
