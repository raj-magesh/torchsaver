from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, StackDataset


def create_image_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    preprocess_fn: Callable[[Image.Image], torch.Tensor],
    indices: list[str],
) -> DataLoader:
    """Create a PyTorch dataloader for loading images and preprocessing them.

    Args:
    ----
        dataset: source dataset that maps a key to a PIL.Image.Image
        preprocess_fn: function used to preprocess each PIL.Image.Image
        batch_size: batch size

    Returns:
    -------
        torch dataloader that produces batches of data in the form (image_tensor, image_id)

    """

    def collate_fn(
        batch: list[tuple[Image.Image, str]],
    ) -> tuple[torch.Tensor, npt.NDArray[np.str_]]:
        images = torch.stack([preprocess_fn(pair[0]) for pair in batch])
        ids = np.array([pair[1] for pair in batch])
        return images, ids

    return DataLoader(
        StackDataset(dataset, dict(zip(indices, indices, strict=True))),
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=indices,
    )
