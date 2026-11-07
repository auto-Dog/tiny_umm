from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import pytest


@pytest.fixture
def artifact_dir():
    """Use a workspace directory so Windows sandboxes can read the test outputs."""
    workspace = Path(__file__).resolve().parents[1]
    root = workspace / "artifacts" / "test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    if not root.is_relative_to(workspace):
        raise RuntimeError("test artifacts must stay inside the workspace")
    path = root / uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent == root and not path.is_symlink():
            rmtree(path)
