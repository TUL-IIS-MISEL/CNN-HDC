# %%
import torch
import h5py
import cv2

import numpy as np

from tqdm import tqdm

# %%
from Datasets.KAIST import KaistPedDataset

# %%
class KaistPedObjectDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str = "/datasets/KAIST_rgbt-ped-detection/data/kaist-rgbt",
        split: str = "train",
    ) -> None:
        super().__init__()
        self.path = path

        if split not in {"train", "test"}:
            raise ValueError
        self.split = split

        self.hdf5_name = "KAIST_PED_OBJECT"

    def _preprocess(self, source_train: KaistPedDataset, source_test: KaistPedDataset):
        count_train = 0
        count_test = 0
        for idx in tqdm(range(len(source_train))):
            _, _, annotations, _ = source_train[idx]
            count_train += len(annotations)
        for idx in tqdm(range(len(source_test))):
            _, _, annotations, _ = source_test[idx]
            count_test += len(annotations)
        print(f"TRAIN: {count_train}\nTEST: {count_test}")

        # count_train = 53293
        # count_test = 54839

        with h5py.File(f"{self.path}/{self.hdf5_name}.h5", "w") as file:
            visible_group = file.create_group("visible")
            infrared_group = file.create_group("infrared")
            label_group = file.create_group("labels")

            visible_train = visible_group.create_dataset(
                "train", (count_train, 32, 32, 3), h5py.h5t.STD_U8LE
            )
            visible_test = visible_group.create_dataset(
                "test", (count_test, 32, 32, 3), h5py.h5t.STD_U8LE
            )

            infrared_train = infrared_group.create_dataset(
                "train", (count_train, 32, 32, 3), h5py.h5t.STD_U8LE
            )
            infrared_test = infrared_group.create_dataset(
                "test", (count_test, 32, 32, 3), h5py.h5t.STD_U8LE
            )

            label_train = label_group.create_dataset(
                "train", (count_train, 12), h5py.h5t.STD_U16LE
            )
            label_test = label_group.create_dataset(
                "test", (count_test, 12), h5py.h5t.STD_U16LE
            )

            pbar = tqdm(total=count_train + count_test)
            for idx, (vis, infra, label) in enumerate(
                self._extract_objects(source_train)
            ):
                visible_train[idx] = vis
                infrared_train[idx] = infra
                label_train[idx] = label
                pbar.update(1)

            for idx, (vis, infra, label) in enumerate(
                self._extract_objects(source_test)
            ):
                visible_test[idx] = vis
                infrared_test[idx] = infra
                label_test[idx] = label
                pbar.update(1)
            pbar.close()

    def _make_square(
        self, x: int, y: int, width: int, height: int, x_max: int, y_max: int
    ):
        assert height <= y_max
        assert width <= x_max
        assert x >= 0
        assert y >= 0

        step = abs(width - height) // 2
        if width > height:
            height = width
            y = min(max(0, y - step), y_max - height)
        else:  # width < height:
            width = height
            x = min(max(0, x - step), x_max - height)
        return x, y, width, height

    def _standardise_shape(self, square, side):
        square = cv2.resize(square, (side, side))
        square *= 255
        square = square.astype(np.uint8)
        return square

    def _extract_objects(self, source_dataset, side: int = 32):
        for idx in range(len(source_dataset)):
            visible, infrared, annotations, _ = source_dataset[idx]
            for annotation in annotations:
                x, y, width, height = self._make_square(
                    annotation[1],
                    annotation[2],
                    annotation[3],
                    annotation[4],
                    512,
                    640,
                )

                # label = annotation[0]
                vis = self._standardise_shape(
                    visible[y : y + height, x : x + width], side
                )
                infra = self._standardise_shape(
                    infrared[y : y + height, x : x + width], side
                )

                yield vis, infra, annotation

    def __len__(self):
        datafile = h5py.File(f"{self.path}/{self.hdf5_name}.h5", "r")
        return len(datafile["visible"][self.split])

    def __getitem__(self, idx: int):
        datafile = h5py.File(f"{self.path}/{self.hdf5_name}.h5", "r")
        return (
            datafile[f"visible/{self.split}"][idx],
            datafile[f"infrared/{self.split}"][idx],
            datafile[f"labels/{self.split}"][idx][0],
        )


# %%
if __name__ == "__main__":
    source_dataset_train = KaistPedDataset(split="train")
    source_dataset_test = KaistPedDataset(split="test")
    target_dataset = KaistPedObjectDataset()
    target_dataset._preprocess(source_dataset_train, source_dataset_test)
