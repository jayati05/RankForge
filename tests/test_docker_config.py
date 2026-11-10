"""Packaging contracts are checked offline; real Docker tests are opt-in."""

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.main import create_app
from app.core.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "docker" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_source_only_and_nonroot():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "@sha256:" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert 'CMD ["python", "docker/entrypoint.py"]' in dockerfile
    assert "COPY . " not in dockerfile
    assert "OPENBLAS_NUM_THREADS" not in dockerfile
    assert "HEALTHCHECK" in dockerfile and "STOPSIGNAL SIGTERM" in dockerfile
    assert "--no-deps" in dockerfile and "pip check" in dockerfile


def test_context_allowlist():
    lines = (ROOT / ".dockerignore").read_text().splitlines()
    assert "**" in lines
    assert {"!app/", "!app/**/*.py", "!docker/requirements.lock"} <= set(lines)
    for path in (".git/", ".venv/", "data/", "experiments/", "models/", "indexes/", ".env*"):
        assert path in lines
        assert "!" + path not in lines


def test_runtime_lock_tracks_declared_runtime_extras():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    lock = (ROOT / "docker/requirements.lock").read_text()
    declared = (
        project["dependencies"]
        + project["optional-dependencies"]["api"]
        + project["optional-dependencies"]["neural"]
    )
    for dependency in declared:
        if dependency.startswith("torch=="):
            assert "torch-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl" in lock
        else:
            assert dependency in lock


def test_mounts_and_shutdown_contract():
    import yaml

    config = yaml.safe_load((ROOT / "compose.yaml").read_text())["services"]["api"]
    assert config["restart"] == "no"
    assert config["read_only"] is True
    assert config["stop_grace_period"] == "60s"
    assert len(config["volumes"]) == 4
    assert all(v["read_only"] and not v["bind"]["create_host_path"] for v in config["volumes"])
    assert "127.0.0.1:" in config["ports"][0]
    assert "OPENBLAS_NUM_THREADS" not in config["environment"]
    assert config["healthcheck"]["test"] == ["CMD", "python", "docker/healthcheck.py"]


def test_env_paths_and_defaults(monkeypatch, tmp_path):
    for key in Settings.model_fields:
        monkeypatch.delenv("RANKFORGE_" + key.upper(), raising=False)
    assert Settings.from_env().default_method == "bm25"
    assert Settings.from_env().threads == 4
    for key in ("dataset_dir", "dense_model_dir", "rerank_model_dir", "dense_index", "policy_dir"):
        monkeypatch.setenv("RANKFORGE_" + key.upper(), str(tmp_path / key))
    monkeypatch.setenv("RANKFORGE_LOG_LEVEL", "warning")
    settings = Settings.from_env()
    assert settings.dataset_dir == tmp_path / "dataset_dir"
    assert settings.log_level == "warning"
    monkeypatch.setenv("RANKFORGE_DEFAULT_METHOD", "unapproved")
    with pytest.raises(ValidationError):
        Settings.from_env()


def test_container_preflight_missing_and_mounted_files(tmp_path, monkeypatch):
    module = load("entrypoint")
    settings = Settings(
        dataset_dir=tmp_path / "dataset",
        dense_model_dir=tmp_path / "dense",
        rerank_model_dir=tmp_path / "rerank",
        dense_index=tmp_path / "index.npz",
        policy_dir=tmp_path / "policy",
    )
    monkeypatch.setattr(module.Settings, "from_env", lambda: settings)
    with pytest.raises(SystemExit, match="Required container resources missing"):
        module.main()
    for path in module.required_files(settings):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    calls = []
    monkeypatch.setattr(module.os, "execv", lambda *args: calls.append(args))
    module.main()
    assert calls[0][1][-2:] == ["-m", "app.api"]


def test_configured_default_method_keeps_explicit_requests():
    settings = Settings(dataset_dir=ROOT / "data/smoke", default_method="dense", default_top_k=3)
    with TestClient(create_app(settings)) as client:
        assert client.post("/search", json={"query": "ERR_42"}).status_code == 503
        response = client.post("/search", json={"query": "ERR_42", "method": "bm25"})
        assert response.status_code == 200 and response.json()["method"] == "bm25"
        assert client.post("/search", json={"query": "ERR_42", "method": None}).status_code == 422


def test_healthcheck_rejects_degraded(monkeypatch):
    import io

    module = load("healthcheck")

    class Opener:
        def open(self, url, timeout):
            assert url.endswith(":8129/health") and timeout == 3
            return io.BytesIO(b'{"status":"degraded"}')

    monkeypatch.setenv("RANKFORGE_PORT", "8129")
    monkeypatch.setattr(module, "build_opener", lambda *args: Opener())
    with pytest.raises(SystemExit):
        module.main()
