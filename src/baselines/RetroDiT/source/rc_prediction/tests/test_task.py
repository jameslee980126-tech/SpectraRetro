import pytest
import torch

from torchdrug_pred.task import (
    multi_positive_accuracy,
    multi_positive_cross_entropy,
    multi_positive_margin_loss,
)


def test_multi_positive_listwise_loss_accepts_any_positive() -> None:
    prediction = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    target = torch.tensor([0, 1, 1])
    size = torch.tensor([3])
    loss = multi_positive_cross_entropy(prediction, target, size)
    expected = torch.logsumexp(prediction, 0) - torch.logsumexp(prediction[1:], 0)
    assert loss.detach().item() == pytest.approx(
        expected.detach().item(),
        abs=1e-6,
    )
    assert multi_positive_accuracy(prediction, target, size) == 1
    loss.backward()
    assert prediction.grad is not None


def test_multi_positive_loss_rejects_missing_positive() -> None:
    loss = multi_positive_cross_entropy(
        torch.tensor([0.0, 1.0]),
        torch.tensor([0, 0]),
        torch.tensor([2]),
    )
    assert torch.isfinite(loss)
    assert loss > 80


def test_hard_negative_margin_only_penalizes_violating_graphs() -> None:
    prediction = torch.tensor([2.0, 0.0, 0.5, 1.0, 1.4])
    target = torch.tensor([1, 0, 0, 1, 0])
    size = torch.tensor([3, 2])
    loss = multi_positive_margin_loss(prediction, target, size, margin=1.0)
    # First graph has no violation. Second graph has 1.4 - 1.0 + 1.0.
    assert loss.item() == pytest.approx(0.7)
