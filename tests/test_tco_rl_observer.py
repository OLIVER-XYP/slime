import json

from slime_plugins.tco_rl import SlimeTCOObserver, get_global_observer


def test_observer_emits_ordered_contract_events(tmp_path):
    path = tmp_path / "trace.jsonl"
    observer = SlimeTCOObserver(path=path, experiment_id="fixture")

    observer.record_rewards(3, [1.0, 0.0])
    observer.record_advantages(3, [0.5, -0.5])
    observer.record_loss(3, loss=0.12, mask_sum=2.0)
    observer.record_optimizer(3, step=1, parameter_version="v1")
    observer.flush()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["stage"] for record in records] == [
        "reward.verify",
        "advantage.compute",
        "grpo.loss",
        "optimizer.step",
    ]
    assert [record["order"] for record in records] == [0, 1, 2, 3]
    assert records[0]["contract_id"] == "rl.reward.verifier_consistency"
    assert records[-1]["lineage"]["parameter_version"] == "v1"


def test_observer_records_policy_and_backward_evidence(tmp_path):
    path = tmp_path / "trace.jsonl"
    observer = SlimeTCOObserver(path=path, experiment_id="fixture")

    observer.record_policy("r0", old_log_probs=[-1.0, -2.0], current_log_probs=[-1.1, -1.8])
    observer.record_backward("r0", grad_norm=0.25)
    observer.flush()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["stage"] for record in records] == ["policy.logprob", "backward.compute"]
    assert records[0]["writes"]["token_count"] == 2
    assert records[1]["contract_id"] == "rl.backward.gradient_finite"


def test_observer_flattens_batched_policy_values_and_lineage(tmp_path):
    path = tmp_path / "trace.jsonl"
    observer = SlimeTCOObserver(path=path, experiment_id="fixture")

    observer.record_policy(
        ["r0", "r0"],
        old_log_probs=[[-1.0, -2.0], [-3.0]],
        current_log_probs=[[-1.1, -1.8], [-2.5]],
    )
    observer.flush()

    record = json.loads(path.read_text().splitlines()[0])
    assert record["lineage"]["rollout_id"] == "r0"
    assert record["writes"]["token_count"] == 3


def test_global_observer_is_disabled_without_trace_path(monkeypatch, tmp_path):
    monkeypatch.delenv("TCO_RL_TRACE_PATH", raising=False)
    assert get_global_observer() is None

    path = tmp_path / "disabled.jsonl"
    assert not path.exists()
