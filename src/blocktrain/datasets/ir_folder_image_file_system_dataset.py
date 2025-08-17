import json
from pathlib import Path

import cv2
import torch

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
        annotations_file : Path,
        image_dir : Path,
        extension:str = "png",
    ):
        """

        Args:
            annotations_file: path to the annotations file
            image_dir: path to the images
            extension: the extension to load, assumes same, specified without dot (e.g. specify png, not .png)
        """
        annotations_file = Path(annotations_file) if isinstance(annotations_file, str) else annotations_file
        image_dir = Path(image_dir) if isinstance(image_dir, str) else image_dir

        self.annotations_file = annotations_file
        self.image_dir = image_dir
        self.extension = extension

    def __len__(self) -> int:
        return len(list(self.image_dir.glob(f"*.{self.extension}")))

    def __getitem__(self, idx):
        labels = json.load(self.annotations_file.open())
        img_path = list(self.image_dir.glob(f"*.{self.extension}"))[idx]
        img = cv2.imread(str(img_path))
        return (
            torch.from_numpy(img), 
            torch.Tensor((
                labels["exist"][idx], 
                *labels["gt_rect"][idx]
            ))
        ) 