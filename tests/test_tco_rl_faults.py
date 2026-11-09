from slime_plugins.tco_rl.faults import apply_reward_fault


def test_reward_fault_is_opt_in_and_single_sample():
    assert apply_reward_fault(1.0, sample_index=0, fault="none") == (1.0, False)
    assert apply_reward_fault(1.0, sample_index=1, fault="reward_flip") == (1.0, False)
    assert apply_reward_fault(1.0, sample_index=0, fault="reward_flip") == (0.0, True)
    assert apply_reward_fault(0.0, sample_index=0, fault="reward_flip", already_applied=True) == (0.0, True)
