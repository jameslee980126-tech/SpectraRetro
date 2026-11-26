from torchdrug_pred.training import _root_selection_key


def test_root_selection_key_uses_documented_tie_breakers() -> None:
    metrics = {
        "root_hit_at_5": 0.98,
        "root_mrr": 0.89,
        "root_hit_at_3": 0.95,
    }
    assert _root_selection_key(metrics) == (0.98, 0.89, 0.95)
