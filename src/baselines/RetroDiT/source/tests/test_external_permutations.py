import json

from src.data.dataset import DatasetModule


def loader() -> DatasetModule:
    return DatasetModule.__new__(DatasetModule)


def test_loads_direct_torchdrug_export(tmp_path) -> None:
    path = tmp_path / "direct.json"
    path.write_text(json.dumps({"CCO": {"0_1_2": 0.7, "1_0_2": 0.3}}))
    values, names = loader()._load_external_permutations(str(path))
    assert names == ["pred_perm"]
    assert values["pred_perm"]["CCO"]["0_1_2"] == 0.7


def test_loads_legacy_wrapped_export(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"gnn_pred_rc": {"CCO": {"0_1_2": 1.0}}}))
    values, names = loader()._load_external_permutations(str(path))
    assert names == ["gnn_pred_rc"]
    assert values["gnn_pred_rc"]["CCO"]["0_1_2"] == 1.0
