from unittest.mock import patch

from src.data.datamodule import ExactDistributedTestSampler


def sampler_indices(length: int, rank: int, world_size: int) -> list[int]:
    dataset = list(range(length))
    with (
        patch("src.data.datamodule.dist.is_available", return_value=True),
        patch("src.data.datamodule.dist.is_initialized", return_value=True),
        patch("src.data.datamodule.dist.get_rank", return_value=rank),
        patch("src.data.datamodule.dist.get_world_size", return_value=world_size),
    ):
        sampler = ExactDistributedTestSampler(dataset)
    indices = list(sampler)
    assert len(indices) == len(sampler)
    return indices


def test_single_rank_is_complete_and_ordered() -> None:
    assert sampler_indices(7, rank=0, world_size=1) == list(range(7))


def test_distributed_shards_are_exact_without_padding() -> None:
    shards = [sampler_indices(10, rank=rank, world_size=3) for rank in range(3)]
    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(10))
    assert len(flattened) == len(set(flattened))


def test_more_ranks_than_samples_produces_empty_shards() -> None:
    shards = [sampler_indices(2, rank=rank, world_size=4) for rank in range(4)]
    assert shards == [[0], [1], [], []]
