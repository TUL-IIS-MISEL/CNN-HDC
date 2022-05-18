import os
from typing import Dict

import torch
from tqdm import tqdm
import imageio
import numpy as np

from Datasets.ViratDataset import ViratDataset, ViratObjectType

import cv2

import h5py


class ViratEventClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path="/datasets/VIRAT_squares") -> None:
        super().__init__()
        self.path = path
        self.list_file = "list_samples.txt"
        self.data: Dict[str, ViratObjectType] = dict()

    def _find_data(self):
        with open(f"{self.path}/{self.list_file}") as listfile:
            for line in tqdm(listfile):
                path, label = line.strip().split(";")
                self.data[path] = int(label.strip())

    def _make_square(self, x, y, width, height, x_max, y_max):
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

    def _preprocess_dataset(
        self,
        virat_dataset: ViratDataset,
        shrinkage: int = 4,
        side=100,
        std_thr=12,
        preview=False,
    ):
        os.makedirs(self.path, exist_ok=True)
        file_list = []
        for idx in tqdm(range(len(virat_dataset))):
            image, objects = virat_dataset[idx]
            img = np.transpose(image.numpy(), (1, 2, 0)).squeeze()
            if preview:
                view = img.copy()
            for odx, obj in enumerate(objects):
                if (
                    obj.object_type != ViratObjectType.CAR
                    and obj.object_type != ViratObjectType.PERSON
                ):
                    continue
                x, y, width, height = self._make_square(
                    obj.bbox_lefttop_x // shrinkage,
                    obj.bbox_lefttop_y // shrinkage,
                    obj.bbox_width // shrinkage,
                    obj.bbox_height // shrinkage,
                    img.shape[1],
                    img.shape[0],
                )
                square = img[y : y + height, x : x + width]
                square = cv2.resize(square, (side, side))
                square *= 255
                square = square.astype(np.uint8)
                std = square.std()
                if std >= std_thr:  # parameter chosen experimentally
                    file_name = f"img_{idx:08d}_{odx:04d}.png"
                    imageio.imwrite(
                        f"{self.path}/{file_name}",
                        square,
                        format="png",
                        compress_level=0,
                    )
                    file_list.append(f"{file_name};{obj.object_type.value:02d}\n")

                if preview:
                    cv2.rectangle(
                        view,
                        (x, y),
                        (x + width, y + height,),
                        {ViratObjectType.CAR: 1, ViratObjectType.PERSON: 0,}.get(
                            obj.object_type, 0.75
                        ),
                    )
                    cv2.putText(
                        view,
                        f"{std: .2f}",
                        (x, y + height),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        1 if std > std_thr else 0,
                    )
            if preview:
                cv2.imshow("", view)
                cv2.setWindowTitle("", f"{idx}")
                cv2.waitKey(1)

        with open(f"{self.path}/{self.list_file}", "w") as outfile:
            for line in file_list:
                outfile.write(line)

    def _convert_to_hdf5(self):
        self._find_data()
        images = []
        labels = []
        for key in tqdm(self.data.keys()):
            images.append(imageio.imread(f"{self.path}/{key}"))
            labels.append(self.data[key])
        images = np.array(images)
        labels = np.array(labels)
        file = h5py.File(f"{self.path}.h5", "w")
        dataset = file.create_dataset(
            "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
        )
        labelset = file.create_dataset(
            "labels", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
        )
        file.close()

    def __len__(self):
        datafile = h5py.File(f"{self.path}.h5", "r")
        return len(datafile["labels"])

    def __getitem__(self, idx):
        datafile = h5py.File(f"{self.path}.h5", "r")
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return datafile["images"][idx], datafile["labels"][idx]


if __name__ == "__main__":
    source_dataset = ViratDataset(initialize=False)
    source_dataset.event_videos_dir = "videos_event_grey_VGA_8"
    source_dataset._find_frames()

    dataset = ViratEventClassificationDataset(
        path="/datasets/VIRAT_squares_VGA_32"
    )
    dataset._preprocess_dataset(source_dataset, preview=True, side=32, shrinkage=3)
    dataset._convert_to_hdf5()

    from collections import Counter

    dataset = ViratEventClassificationDataset(
        path="/datasets/VIRAT_squares_VGA_32"
    )
    print(Counter(h5py.File(f"{dataset.path}.h5", "r")["labels"]))
