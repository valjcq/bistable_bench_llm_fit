"""scripts/pilot.sh must be safe to run wrongly: no server contact, no shell poisoning. These tests
only exercise its early exits and the --make-config extraction (in a scratch copy)."""

import shutil
import subprocess
from pathlib import Path

import yaml

from bench.config import AgentConfig

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: str, cwd: Path = ROOT, env=None):
    import os

    environ = {k: v for k, v in os.environ.items() if k != "AGENT"} | (env or {})
    return subprocess.run(["bash", "-c", cmd], cwd=cwd, env=environ, capture_output=True, text=True)


def test_without_agent_it_explains_and_exits_1_before_running_anything():
    r = run("bash scripts/pilot.sh")
    assert r.returncode == 1 and "--make-config" in r.stderr and "runs/pilot" not in r.stdout
    r = run("bash scripts/pilot.sh", env={"AGENT": "configs/does_not_exist.yaml"})
    assert r.returncode == 1 and "does_not_exist" in r.stderr


def test_sourcing_it_is_refused_without_closing_or_poisoning_the_shell():
    r = run(". scripts/pilot.sh; echo still-alive; case $- in *e*) echo errexit;; esac")
    assert "do not source" in r.stderr and "still-alive" in r.stdout
    assert "errexit" not in r.stdout  # `set -e` was not switched on in the caller


def test_make_config_extracts_a_valid_agent_section(tmp_path):
    for name in ("scripts/pilot.sh", "configs/agent_llamacpp_example.yaml"):
        (tmp_path / Path(name).parent).mkdir(exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    r = run("bash scripts/pilot.sh --make-config", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    raw = yaml.safe_load((tmp_path / "configs/agent_local.yaml").read_text())
    cfg = AgentConfig(**raw["agent"])  # the same validation the runner applies
    assert cfg.backend == "openai_compatible" and cfg.base_url == "http://localhost:8001/v1"
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.min_p) == (0.6, 0.95, 20, 0.0)
    assert cfg.expected_model == "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" and cfg.api_key_env is None
    assert run("bash scripts/pilot.sh --make-config", cwd=tmp_path).returncode == 1  # no overwrite
