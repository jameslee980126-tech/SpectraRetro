from unittest.mock import patch

from torchdrug_pred.cli import _load_solver, build_parser


def test_training_defaults_select_on_validation_roots_and_export_valid() -> None:
    args = build_parser().parse_args(
        [
            "train_eval_export",
            "--train-path", "train.pt",
            "--valid-path", "valid.pt",
            "--test-path", "test.pt",
            "--output-dir", "out",
        ]
    )
    assert args.export_splits == ["valid"]
    assert args.selection_center_top_k == 5
    assert args.selection_root_limit == 5
    assert args.short_cut is False
    assert args.batch_norm is False
    assert args.concat_hidden is True
    assert args.objective == "single_positive"
    assert args.temperature == [1.0]
    assert args.root_aggregation == ["current"]
    assert args.export_policies == ["reference", "strict_root"]
    assert args.ranking_loss_weight == 0.0
    assert args.ranking_margin == 1.0


def test_export_policy_can_select_strict_only() -> None:
    args = build_parser().parse_args(
        [
            "export",
            "--train-path", "train.pt",
            "--valid-path", "valid.pt",
            "--test-path", "test.pt",
            "--output-dir", "out",
            "--checkpoint", "best.pt",
            "--export-policies", "strict_root",
        ]
    )
    assert args.export_policies == ["strict_root"]


def test_structure_flags_are_recordable() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--train-path", "train.pt",
            "--valid-path", "valid.pt",
            "--test-path", "test.pt",
            "--output-dir", "out",
            "--short-cut",
            "--batch-norm",
            "--dropout", "0.1",
            "--no-concat-hidden",
            "--readout", "mean",
        ]
    )
    assert args.short_cut is True
    assert args.batch_norm is True
    assert args.dropout == 0.1
    assert args.concat_hidden is False
    assert args.readout == "mean"


def test_multi_positive_objective_is_recordable() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--train-path", "train.pt",
            "--valid-path", "valid.pt",
            "--test-path", "test.pt",
            "--output-dir", "out",
            "--objective", "multi_positive",
        ]
    )
    assert args.objective == "multi_positive"


def test_margin_arguments_are_passed_to_solver_not_dataset_loader() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--train-path", "train.pt",
            "--valid-path", "valid.pt",
            "--test-path", "test.pt",
            "--output-dir", "out",
            "--objective", "multi_positive_margin",
            "--ranking-loss-weight", "0.25",
            "--ranking-margin", "1.0",
        ]
    )
    with (
        patch("torchdrug_pred.training.set_seed"),
        patch(
            "torchdrug_pred.training.load_datasets",
            return_value=("train", "valid", "test"),
        ) as load,
        patch("torchdrug_pred.training.build_solver", return_value="solver") as build,
    ):
        solver, *_ = _load_solver(args)
    assert solver == "solver"
    assert load.call_args.args[-1] == "multi_positive_margin"
    assert len(load.call_args.args) == 6
    assert build.call_args.args[-3:] == (
        "multi_positive_margin",
        0.25,
        1.0,
    )
