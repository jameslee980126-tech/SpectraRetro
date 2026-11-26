import json

import pytest

from torchdrug_pred.export import (
    CenterCandidate,
    ProductPrediction,
    aggregate_root_scores,
    build_permutation_info,
    calculate_ranking_metrics,
    export_prediction_sets,
)
from torchdrug_pred.io import (
    add_permutation_score,
    validate_permutation_info,
    write_permutation_json,
)


PRODUCT = "[CH3:1][CH2:2][OH:3]"
REACTION = f"[CH3:1][CH2:2][OH:3]>>{PRODUCT}"


def candidates() -> tuple[CenterCandidate, ...]:
    return (
        CenterCandidate(
            atom_indices=(1, 2),
            atom_maps=(2, 3),
            log_likelihood=-0.1,
        ),
        CenterCandidate(
            atom_indices=(0,),
            atom_maps=(1,),
            log_likelihood=-1.0,
        ),
        CenterCandidate(
            atom_indices=(1,),
            atom_maps=(2,),
            log_likelihood=-2.0,
        ),
    )


def test_edge_probability_is_added_to_both_roots() -> None:
    roots = aggregate_root_scores(candidates(), 2, 1.0, 3)
    scores = dict(roots)
    assert list(scores)[:2] == [1, 2]
    assert scores[1] == pytest.approx(scores[2])
    assert sum(scores.values()) == pytest.approx(1.0)


def test_temperature_and_strict_root_limit() -> None:
    cold = aggregate_root_scores(candidates(), 3, 0.5, 2)
    hot = aggregate_root_scores(candidates(), 3, 2.0, 2)
    assert len(cold) == 2
    assert len(hot) == 2
    assert cold[0][1] > hot[0][1]


def test_root_aggregation_variants_change_edge_mass() -> None:
    current = dict(aggregate_root_scores(candidates(), 2, 1.0, 3, "current"))
    split = dict(aggregate_root_scores(candidates(), 2, 1.0, 3, "split_edge"))
    reciprocal = aggregate_root_scores(
        candidates(), 3, 1.0, 3, "reciprocal_rank"
    )
    hybrid = aggregate_root_scores(
        candidates(), 3, 1.0, 3, "probability_rank_hybrid"
    )
    assert current[1] > split[1]
    assert sum(split.values()) == pytest.approx(1.0)
    assert sum(score for _, score in reciprocal) == pytest.approx(1.0)
    assert sum(score for _, score in hybrid) == pytest.approx(1.0)


def test_build_permutation_info_and_identity_fallback() -> None:
    predictions = [
        ProductPrediction(
            product_smiles=PRODUCT,
            true_atom_maps=(2, 3),
            candidates=candidates(),
        )
    ]
    permutation_info, metrics = build_permutation_info(
        predictions,
        [REACTION],
        center_top_k=2,
        temperature=1.0,
        root_limit=2,
    )
    validate_permutation_info(permutation_info)
    assert metrics["center_hit_at_1"] == 1.0
    assert metrics["root_hit_at_1"] == 1.0
    assert metrics["mean_candidates"] == 2.0
    missing_info, missing_metrics = build_permutation_info(
        [
            ProductPrediction(
                product_smiles=PRODUCT,
                true_atom_maps=(2, 3),
                candidates=(),
            )
        ],
        [REACTION],
        center_top_k=2,
        temperature=1.0,
        root_limit=2,
    )
    validate_permutation_info(missing_info)
    assert missing_metrics["prediction_coverage"] == 0.0
    assert next(iter(missing_info.values())) == {"0_1_2": 1.0}


def test_fast_ranking_metrics_match_export_metrics() -> None:
    predictions = [
        ProductPrediction(
            product_smiles=PRODUCT,
            true_atom_maps=(2, 3),
            candidates=candidates(),
        )
    ]
    _, export_metrics = build_permutation_info(
        predictions, [REACTION], center_top_k=3, temperature=1.0, root_limit=3
    )
    fast_metrics = calculate_ranking_metrics(
        predictions, center_top_k=3, temperature=1.0, root_limit=3
    )
    assert fast_metrics == export_metrics


def test_ranking_accepts_any_multi_positive_center() -> None:
    prediction = ProductPrediction(
        product_smiles=PRODUCT,
        true_atom_maps=(2, 3),
        candidates=candidates(),
        product_atom_maps=(1, 2, 3),
        true_center_atom_maps=((1,), (2, 3)),
    )
    metrics = calculate_ranking_metrics(
        [prediction], center_top_k=3, temperature=1.0, root_limit=3
    )
    assert metrics["center_hit_at_1"] == 1.0
    assert metrics["root_hit_at_1"] == 1.0


def test_duplicate_products_and_symmetric_permutations_merge(tmp_path) -> None:
    prediction = ProductPrediction(
        product_smiles=PRODUCT,
        true_atom_maps=(2, 3),
        candidates=candidates(),
    )
    permutation_info, _ = build_permutation_info(
        [prediction, prediction],
        [REACTION],
        center_top_k=2,
        temperature=1.0,
        root_limit=2,
    )
    assert sum(next(iter(permutation_info.values())).values()) == pytest.approx(2.0)
    symmetric: dict[str, dict[str, float]] = {}
    assert add_permutation_score(symmetric, "[CH3:1][CH3:2]", 0, 0.4)
    assert add_permutation_score(symmetric, "[CH3:1][CH3:2]", 1, 0.6)
    assert sum(next(iter(symmetric.values())).values()) == pytest.approx(1.0)
    output = tmp_path / "predictions.json"
    write_permutation_json(output, symmetric)
    assert json.loads(output.read_text()) == symmetric


def test_graph_root_is_mapped_to_raw_product_atom_order() -> None:
    prediction = ProductPrediction(
        product_smiles="[OH:3][CH2:2][CH3:1]",
        true_atom_maps=(2, 3),
        candidates=candidates(),
        product_atom_maps=(1, 2, 3),
    )
    permutation_info, metrics = build_permutation_info(
        [prediction],
        [f"[CH3:1][CH2:2][OH:3]>>{prediction.product_smiles}"],
        center_top_k=1,
        temperature=1.0,
        root_limit=2,
    )
    validate_permutation_info(permutation_info)
    assert metrics["prediction_coverage"] == 1.0


def test_export_writes_reference_and_strict_sets(tmp_path) -> None:
    prediction = ProductPrediction(
        product_smiles=PRODUCT,
        true_atom_maps=(2, 3),
        candidates=candidates(),
    )
    manifest = export_prediction_sets(
        [prediction],
        [REACTION],
        tmp_path / "predictions",
        "model",
        [3, 5],
        [3, 5],
        10,
        1.0,
    )
    assert len(manifest) == 4
    assert (tmp_path / "predictions" / "index.json").is_file()


def test_export_temperature_aggregation_grid_has_unique_names(tmp_path) -> None:
    prediction = ProductPrediction(
        product_smiles=PRODUCT,
        true_atom_maps=(2, 3),
        candidates=candidates(),
    )
    manifest = export_prediction_sets(
        [prediction],
        [REACTION],
        tmp_path / "grid",
        "model",
        [3],
        [3],
        3,
        [0.5, 1.0],
        aggregation_methods=["current", "split_edge"],
    )
    assert len(manifest) == 8
    assert len({entry["prediction_path"] for entry in manifest.values()}) == 8
    assert all("_temp" in stem and "_agg_" in stem for stem in manifest)


def test_export_can_skip_reference_policy(tmp_path) -> None:
    prediction = ProductPrediction(
        product_smiles=PRODUCT,
        true_atom_maps=(2, 3),
        candidates=candidates(),
    )
    manifest = export_prediction_sets(
        [prediction],
        [REACTION],
        tmp_path / "strict_only",
        "model",
        [3],
        [3],
        10,
        [1.0, 1.5],
        export_policies=["strict_root"],
    )
    assert len(manifest) == 2
    assert all("strict_root" in stem for stem in manifest)


def test_split_manifests_do_not_mix_paths_or_identity_fallbacks(tmp_path) -> None:
    valid_reaction = f"[CH3:1][OH:2]>>[CH3:1][OH:2]"
    test_reaction = f"[CH3:1][NH2:2]>>[CH3:1][NH2:2]"
    valid_manifest = export_prediction_sets(
        [],
        [valid_reaction],
        tmp_path / "predictions" / "valid",
        "model",
        [1],
        [1],
        1,
        1.0,
        split="valid",
        metrics_dir=tmp_path / "metrics" / "valid",
    )
    test_manifest = export_prediction_sets(
        [],
        [test_reaction],
        tmp_path / "predictions" / "test",
        "model",
        [1],
        [1],
        1,
        1.0,
        split="test",
        metrics_dir=tmp_path / "metrics" / "test",
    )
    valid_entry = next(iter(valid_manifest.values()))
    test_entry = next(iter(test_manifest.values()))
    assert valid_entry["split"] == "valid"
    assert test_entry["split"] == "test"
    assert "/valid/" in valid_entry["prediction_path"]
    assert "/test/" in test_entry["prediction_path"]
    valid_data = json.loads(open(valid_entry["prediction_path"]).read())
    test_data = json.loads(open(test_entry["prediction_path"]).read())
    assert set(valid_data) != set(test_data)
    validate_permutation_info(valid_data)
    validate_permutation_info(test_data)
