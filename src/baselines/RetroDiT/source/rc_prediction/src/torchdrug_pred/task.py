from __future__ import annotations

import torch


def multi_positive_cross_entropy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    size: torch.Tensor,
) -> torch.Tensor:
    """Listwise CE where any labeled candidate is a correct outcome."""
    from torchdrug.layers import functional

    positive = target > 0
    log_probability = functional.variadic_log_softmax(prediction, size)
    positive_probability = functional.variadic_sum(
        log_probability.exp() * positive.to(log_probability.dtype),
        size,
    )
    tiny = torch.finfo(prediction.dtype).tiny
    return -positive_probability.clamp_min(tiny).log().mean()


def multi_positive_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    size: torch.Tensor,
) -> torch.Tensor:
    from torchdrug.layers import functional

    local_index = functional.variadic_max(prediction, size)[1]
    starts = size.cumsum(0) - size
    return (target[starts + local_index] > 0).float().mean()


def multi_positive_margin_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    size: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Encourage the best positive to outrank the hardest negative."""
    from torchdrug.layers import functional

    positive = target > 0
    negative_infinity = torch.tensor(
        float("-inf"),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    positive_max = functional.variadic_max(
        prediction.masked_fill(~positive, negative_infinity),
        size,
    )[0]
    negative_max = functional.variadic_max(
        prediction.masked_fill(positive, negative_infinity),
        size,
    )[0]
    losses = torch.relu(negative_max - positive_max + margin)
    return torch.where(torch.isfinite(losses), losses, 0).mean()


def build_multi_positive_task(
    model: torch.nn.Module,
    ranking_loss_weight: float = 0.0,
    ranking_margin: float = 1.0,
):
    from torchdrug import tasks

    if ranking_loss_weight < 0:
        raise ValueError("ranking_loss_weight must be non-negative")
    if ranking_margin <= 0:
        raise ValueError("ranking_margin must be positive")

    class MultiPositiveCenterIdentification(tasks.CenterIdentification):
        def forward(self, batch):
            all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            metric = {}
            prediction = self.predict(batch, all_loss, metric)
            target, size = self.target(batch)
            accuracy = multi_positive_accuracy(prediction, target, size)
            loss = multi_positive_cross_entropy(prediction, target, size)
            if ranking_loss_weight:
                ranking_loss = multi_positive_margin_loss(
                    prediction,
                    target,
                    size,
                    ranking_margin,
                )
                metric["hard negative margin"] = ranking_loss
                loss = loss + ranking_loss_weight * ranking_loss
            metric[tasks._get_metric_name("acc")] = accuracy
            metric[tasks._get_criterion_name("ce")] = loss
            return all_loss + loss, metric

        def evaluate(self, prediction, target):
            labels, size = target
            return {
                tasks._get_metric_name("acc"): multi_positive_accuracy(
                    prediction,
                    labels,
                    size,
                )
            }

    return MultiPositiveCenterIdentification(
        model,
        feature=("graph", "atom", "bond"),
    )
