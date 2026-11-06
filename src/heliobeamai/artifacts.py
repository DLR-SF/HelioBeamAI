"""Released model files: where they live, download and verification.

    from heliobeamai import artifacts
    artifacts.download("prediction")          # transformer, encoder cache, projection CNN (~ 30 MB)
    root = artifacts.root()                   # local folder (see below)

Command line: ``heliobeamai download [prediction|training|all]``.

Local folder (:func:`root`): ``$HELIOBEAMAI_ARTIFACTS`` if set, else
``<repository>/artifacts`` for a source checkout, else
``~/.cache/heliobeamai/artifacts``. Files are stored under the paths of
:data:`MANIFEST` (``transformer/T2026.pt`` ...); on the release page each file
is an asset named ``path.replace("/", "__")``.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

PathLike = Union[str, Path]

RELEASE = "models-v1"
DEFAULT_URL = f"https://github.com/DLR-SF/HelioBeamAI/releases/download/{RELEASE}/"
_REPO_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


@dataclass(frozen=True)
class Artifact:
    path: str          # relative to root()
    sha256: str
    size: int          # bytes
    group: str         # prediction | training
    description: str

    @property
    def asset(self) -> str:
        return self.path.replace("/", "__")


MANIFEST: Dict[str, Artifact] = {}


def _register(path: str, sha256: str, size: int, group: str, description: str) -> None:
    MANIFEST[path] = Artifact(path, sha256, size, group, description)


# -- release models-v1 (generated manifest) -------------------------------------------
from ._manifest import ENTRIES as _ENTRIES  # noqa: E402

for _e in _ENTRIES:
    _register(*_e)

GROUPS = ("prediction", "training")


def root() -> Path:
    """Local artifact folder."""
    env = os.environ.get("HELIOBEAMAI_ARTIFACTS")
    if env:
        return Path(env).expanduser()
    if _REPO_ARTIFACTS.is_dir():
        return _REPO_ARTIFACTS
    return Path.home() / ".cache" / "heliobeamai" / "artifacts"


def select(groups: Union[str, Iterable[str]] = "prediction") -> List[Artifact]:
    groups = [groups] if isinstance(groups, str) else list(groups)
    if "all" in groups:
        groups = list(GROUPS)
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"Unknown artifact groups {sorted(unknown)}; choose from {GROUPS} or 'all'")
    return [a for a in MANIFEST.values() if a.group in groups]


def sha256(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(groups: Union[str, Iterable[str]] = "prediction", root_dir: Optional[PathLike] = None
           ) -> Dict[str, str]:
    """``{path: "ok" | "missing" | "checksum mismatch"}``."""
    base = Path(root_dir) if root_dir else root()
    out = {}
    for a in select(groups):
        f = base / a.path
        out[a.path] = "missing" if not f.exists() else ("ok" if sha256(f) == a.sha256 else "checksum mismatch")
    return out


def download(groups: Union[str, Iterable[str]] = "prediction", root_dir: Optional[PathLike] = None,
             base_url: Optional[str] = None, force: bool = False, quiet: bool = False) -> Path:
    """Download missing files of the given groups and check their SHA-256.

    ``base_url`` defaults to ``$HELIOBEAMAI_ARTIFACTS_URL`` or the GitHub
    release :data:`RELEASE`. Existing files with the right checksum are kept.
    """
    base = Path(root_dir) if root_dir else root()
    url = (base_url or os.environ.get("HELIOBEAMAI_ARTIFACTS_URL") or DEFAULT_URL).rstrip("/") + "/"
    for a in select(groups):
        target = base / a.path
        if target.exists() and not force:
            if sha256(target) == a.sha256:
                continue
            raise RuntimeError(f"{target} exists but its checksum differs from release {RELEASE}; "
                               f"remove it or use force=True.")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not quiet:
            print(f"downloading {a.path} ({a.size / 1e6:.1f} MB)")
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            try:
                with urllib.request.urlopen(url + a.asset) as r:
                    shutil.copyfileobj(r, tmp)
            except Exception as exc:
                os.unlink(tmp.name)
                raise RuntimeError(f"Could not download {url + a.asset}: {exc}") from exc
        if sha256(tmp.name) != a.sha256:
            os.unlink(tmp.name)
            raise RuntimeError(f"Checksum mismatch for {a.path} from {url}")
        os.replace(tmp.name, target)
    return base


def require(path: str, root_dir: Optional[PathLike] = None) -> Path:
    """Local path of an artifact; clear error with the download command if missing."""
    f = (Path(root_dir) if root_dir else root()) / path
    if not f.exists():
        group = MANIFEST[path].group if path in MANIFEST else "all"
        raise FileNotFoundError(f"{f} not found. Download the model files with "
                                f"`heliobeamai download {group}` or set $HELIOBEAMAI_ARTIFACTS.")
    return f
