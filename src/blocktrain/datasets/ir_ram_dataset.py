import json
from pathlib import Path

import cv2
from torch.utils.data import Dataset
import torch

from blocktrain.datasets.label import from_ir_json_labels, Label


class FolderImageRAMDataset(Dataset):
    """
    A torch dataset coupled to a folder of the IR training dataset

    Loads from RAM upon __getitem__ call

    implementation inspired by pytorch docs:
    https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

    def __init__(
        self,
        annotations_file: Path,
        image_dir: Path,
        extension: str = "png",
    ):
        """

        Args:
            annotations_file: path to the annotations file
            image_dir: path to the images
            extension: the extension to load, assumes same, specified without dot (e.g. specify png, not .png)
        """
        annotations_file = Path(annotations_file) if isinstance(annotations_file, str) else annotations_file
        image_dir = Path(image_dir) if isinstance(image_dir, str) else image_dir

        self.num_images: int = len(list(image_dir.glob(f"*.{extension}")))

        self.images: list[cv2.typing.MatLike] = [
            cv2.imread(str(img_path))
            for img_path in list(image_dir.glob(f"*.{extension}"))
        ]

        annotations: dict = json.load(annotations_file.open())

        self.labels: list[Label] = [
            from_ir_json_labels(annotations, idx)
            for idx in range(0, self.num_images)
        ]

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.images[idx]),
            torch.Tensor((
                self.labels[idx].exist,
                *[
                    self.labels[idx].x,
                    self.labels[idx].y,
                    self.labels[idx].width,
                    self.labels[idx].height,
                ]
            ))
        )