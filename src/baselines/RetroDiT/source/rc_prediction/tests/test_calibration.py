import pytest

from torchdrug_pred.io import calibrate_permutation_scores


def test_hot_temperature_flattens_scores_without_changing_order() -> None:
    prediction = {"CC": {"0_1": 0.9, "1_0": 0.1}}
    calibrated = calibrate_permutation_scores(prediction, 2.0)
    assert list(calibrated["CC"]) == ["0_1", "1_0"]
    assert calibrated["CC"]["0_1"] < 0.9
    assert sum(calibrated["CC"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf")])
def test_temperature_must_be_finite_and_positive(temperature: float) -> None:
    with pytest.raises(ValueError):
        calibrate_permutation_scores({"CC": {"0_1": 1.0}}, temperature)
