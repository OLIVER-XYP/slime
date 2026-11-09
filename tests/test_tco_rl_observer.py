import json

from slime_plugins.tco_rl import SlimeTCOObserver


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
