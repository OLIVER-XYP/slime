import json
import os

from slime_plugins.tco_rl.observer import SlimeTCOObserver


def test_observer_replaces_placeholders_without_format_parsing(monkeypatch, tmp_path):
    path = tmp_path / "trace-{role}-{pid}.jsonl"
    monkeypatch.setenv("TCO_RL_TRACE_PATH", str(path))
    monkeypatch.setenv("TCO_RL_ROLE", "rollout")

    observer = SlimeTCOObserver.from_env(experiment_id="fixture")
    observer.emit(stage="rollout.generate", operator="fixture")
    observer.flush()

    # The exact PID is intentionally read from the generated directory rather
    # than reconstructed in the assertion; the key check is that formatting
    # did not raise and a trace was written.
    files = list(tmp_path.glob(f"trace-rollout-{os.getpid()}.jsonl"))
    assert files
    assert json.loads(files[0].read_text().splitlines()[0])["stage"] == "rollout.generate"


def test_observer_keeps_malformed_template_literal(monkeypatch, tmp_path):
    path = tmp_path / "trace-{role.jsonl"
    monkeypatch.setenv("TCO_RL_TRACE_PATH", str(path))

    observer = SlimeTCOObserver.from_env(experiment_id="fixture")
    observer.emit(stage="reward.verify", operator="fixture")
    observer.flush()

    assert path.exists()
