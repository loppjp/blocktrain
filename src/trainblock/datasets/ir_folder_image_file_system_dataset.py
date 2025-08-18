import json
from pathlib import Path

import cv2
import torch
from torchvision import tv_tensors

from torch.utils.data import Dataset


class FolderImageFileSystemDataset(Dataset):
    """
    A torch dataset coupled to a folder of the IR training dataset

    Loads from filesystem upon __getitem__ call

    implementation inspired by pytorch docs:
    https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

    def __init__(
        self,
        annotations_file: Path,
        image_dir: Path,
        extension: str = "png",
        format_str: str = "XYWH",
        #format_str: str = "CXCYWH",
    ):
        """

        Args:
            annotations_file: path to the annotations file
            image_dir: path to the images
            extension: the extension to load, assumes same, specified without dot (e.g. specify png, not .png)
        """
        annotations_file = (
            Path(annotations_file)
            if isinstance(annotations_file, str)
            else annotations_file
        )
        image_dir = Path(image_dir) if isinstance(image_dir, str) else image_dir

        self.annotations_file = annotations_file
        self.image_dir = image_dir
        self.extension = extension
        self.format = tv_tensors.BoundingBoxFormat(format_str)

        self.labels = json.load(self.annotations_file.open())

    def __len__(self) -> int:
        return len([(e, b) for e, b in zip(self.labels["exist"], self.labels["gt_rect"]) if e == 1 or sum(b) > 0.0])

    def __getitem__(self, idx):
        img_path = list(self.image_dir.glob(f"*.{self.extension}"))[idx]
        img = cv2.imread(str(img_path))
        box = tv_tensors.BoundingBoxes(
            [
                self.labels["gt_rect"][idx][2],
                self.labels["gt_rect"][idx][3],
                self.labels["gt_rect"][idx][0],
                self.labels["gt_rect"][idx][1],
            ],
            format=self.format,
            canvas_size=img.shape[:-1],
            dtype=int
        )
        return (tv_tensors.Image(img.transpose([2, 0, 1]), dtype=torch.double), box.reshape([1, 4]))
