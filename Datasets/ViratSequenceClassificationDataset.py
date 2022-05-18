from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
import numpy as np

from Datasets.ViratDataset import ViratDataset, ViratObjectType

import cv2

import h5py


@dataclass
class SequenceData:
    images: np.ndarray
    label: ViratObjectType


class ViratSequenceClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str = "/datasets/VIRAT_sequences_VGA_32",
        window_size: int = 0,
    ) -> None:
        super().__init__()
        self.path: str = path
        self.label_file: str = "labels.txt"
        self.data: Dict[str, SequenceData] = dict()  # GUID -> data
        self.window_size: int = window_size
        self.sequence_windows: List[
            Tuple[str, np.ndarray, int]
        ] = self._compute_windows() if self.window_size else None

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

    def _preprocess_dataset(
        self,
        source_dataset: ViratDataset,
        shrinkage: int = 3,
        side: int = 32,
        std_thr: int = 12,
    ):
        self._load_images(source_dataset, shrinkage, side, std_thr)

        # remove too short sequences
        guids = list(self.data.keys())
        for guid in guids:
            if len(self.data[guid].images) < 20:
                del self.data[guid]

        self._create_hdf5()

    def _create_hdf5(self):
        key_mapper = {guid: idx for idx, guid in enumerate(sorted(self.data.keys()))}
        object_guids = np.array(sorted(self.data.keys()), dtype="S")
        sequence_lengths = np.zeros((len(object_guids,)), dtype=int)
        for guid in object_guids:
            sequence_lengths[key_mapper[guid.decode()]] = self.data[
                guid.decode()
            ].images.shape[0]
        index_guids = np.repeat(object_guids, sequence_lengths)
        index_offsets = np.concatenate(
            [np.arange(length) for length in sequence_lengths]
        )

        with h5py.File(f"{self.path}.h5", "w") as file:
            meta = file.create_group("metadata")
            meta.create_dataset("object_guids", data=object_guids)
            meta.create_dataset("sequence_lenghts", data=sequence_lengths)
            meta.create_dataset("index_guids", data=index_guids)
            meta.create_dataset("index_offsets", data=index_offsets)

            images = file.create_group("images")
            for guid in object_guids:
                images.create_dataset(
                    guid.decode(), data=self.data[guid.decode()].images
                )

            labels = np.array(
                [self.data[guid.decode()].label.value for guid in object_guids],
                dtype=int,
            )
            labels = np.repeat(labels, sequence_lengths)
            file.create_dataset("labels", data=labels)

    def _load_images(
        self,
        source_dataset: ViratDataset,
        shrinkage: int = 3,
        side: int = 32,
        std_thr: int = 12,
    ):
        for idx in tqdm(range(len(source_dataset))):
            image, objects = source_dataset[idx]
            file_name = sorted(source_dataset.data.keys())[idx]

            img = np.transpose(image.numpy(), (1, 2, 0)).squeeze()

            for _, obj in enumerate(objects):
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
                    guid = f"{file_name.split('/')[0]}OBJ{obj.object_id}"
                    if guid in self.data:
                        self.data[guid].images = np.append(
                            self.data[guid].images,
                            np.expand_dims(square, axis=0),
                            axis=0,
                        )
                    else:
                        self.data[guid] = SequenceData(
                            np.expand_dims(square, axis=0), obj.object_type
                        )

    def _window(self, tensor_length: int, width: int, stride: int):
        windows = (
            torch.arange(start=0, end=width).unsqueeze(0)
            + torch.arange(start=0, end=tensor_length - width, step=stride)
            .unsqueeze(0)
            .T
        )
        return windows

    def _compute_windows(self):
        datafile = h5py.File(f"{self.path}.h5", "r")
        label_offsets = np.cumsum(datafile["metadata"]["sequence_lenghts"])
        label_offsets = np.insert(label_offsets, 0, 0, axis=0)[:-1]
        windows = []
        for guid, length, offset in zip(
            datafile["metadata"]["object_guids"],
            datafile["metadata"]["sequence_lenghts"],
            label_offsets,
        ):
            guid = guid.decode()
            # stride equal to window_size would give non-overlapping windows
            subsequence_windows = self._window(length, self.window_size, 1)
            for window in subsequence_windows:
                label_idx = offset + window[0]
                windows.append((guid, window, label_idx))
        datafile.close()
        return windows

    def __len__(self):
        if self.window_size:
            return len(self.sequence_windows)
        else:
            datafile = h5py.File(f"{self.path}.h5", "r")
            length = len(datafile["labels"])
            datafile.close()
            return length

    def __getitem__(self, idx: int):
        datafile = h5py.File(f"{self.path}.h5", "r")
        if self.window_size:
            guid, window, label_idx = self.sequence_windows[idx]
            images = datafile["images"][guid][window[0] : window[-1] + 1]
            labels = datafile["labels"][label_idx]
        else:
            guid = datafile["metadata"]["index_guids"][idx].decode()
            offset = datafile["metadata"]["index_offsets"][idx]
            images = datafile["images"][guid][offset]
            labels = datafile["labels"][idx]
        datafile.close()
        return images, labels


if __name__ == "__main__":
    source_dataset = ViratDataset(initialize=False)
    source_dataset.event_videos_dir = "videos_event_grey_VGA_8"
    source_dataset._find_frames()

    dataset = ViratSequenceClassificationDataset()
    dataset._preprocess_dataset(source_dataset, side=32, shrinkage=3)

    from collections import Counter

    dataset = ViratSequenceClassificationDataset()
    print(f"Images count: {len(dataset)}")
    print(Counter(h5py.File(f"{dataset.path}.h5", "r")["labels"]))

    dataset = ViratSequenceClassificationDataset(window_size=3)
    print(f"Sequences count: {len(dataset)}")
    print(Counter([guid for guid, _, _ in dataset.sequence_windows]))
