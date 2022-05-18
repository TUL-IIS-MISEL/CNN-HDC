# %%
import os
from distutils.dir_util import mkpath
import dataclasses

# %%
import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import imageio
from tqdm import tqdm

# %%
import sys

sys.path.append("/home/pluczak/MISEL/")

from Datasets.KAIST import KaistPedObject

# %%
class KaistSquaresDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str = "/datasets/KAIST_rgbt-ped-detection/data/kaist-rgbt",
        split: str = "train",
        spectrum: str = "visible",
    ) -> None:
        super().__init__()
        self.path = path

        self.train_name = "train"
        self.test_name = "test"
        if split not in {self.train_name, self.test_name}:
            raise ValueError
        self.split = split

        self.annotations_dir = "annotations"
        self.images_dir = "images"
        self.events_dir = "events"
        self.squares_dir = "squares_THR_08"

        self.visible_dir = "visible"
        self.infrared_dir = "lwir"
        if spectrum not in {self.visible_dir, self.infrared_dir}:
            raise ValueError
        self.spectrum = spectrum

        self.train_suffixes = {"00", "01", "02", "03", "04", "05"}
        self.test_suffixes = {"06", "07", "08", "09", "10", "11"}

        self.kaist_fps = 20
        self.target_frame_time = 4.0 / 30.0
        # maximum distance in pixels between centers of objects in consecutive
        # frames for them to be considered as the same object
        self.center_eps = 16
        self.std_thr = 8
        self.time_thr = 1.5 * self.target_frame_time  # seconds

        self.labels = None

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
        return int(x), int(y), int(width), int(height)

    def _standardise_shape(self, square, side):
        square = cv2.resize(square, (side, side))
        square *= 255
        square = square.astype("uint8")
        return square

    def _is_train_set(self, set_dir_name: str):
        return set_dir_name[-2:] in self.train_suffixes

    def _read_annotations(self, set_dir, vid_dir, annotation_path):
        with open(
            f"{self.path}/{self.annotations_dir}/{set_dir}/{vid_dir}/{annotation_path}"
        ) as annotation_file:
            annotations = np.array(
                [
                    annotation.to_numpy()
                    for annotation in KaistPedObject.parse_file(annotation_file)
                    if annotation
                ]
            )

        return annotations

    def _aggregate_annotations(self):
        dataset_annotations = []
        for set_dir in tqdm(os.listdir(f"{self.path}/{self.annotations_dir}")):
            set_idx = int(set_dir[-2:])
            for vid_dir in os.listdir(f"{self.path}/{self.annotations_dir}/{set_dir}"):
                vid_idx = int(vid_dir[-2:])
                for annotation_path in os.listdir(
                    f"{self.path}/{self.annotations_dir}/{set_dir}/{vid_dir}"
                ):
                    annotation_idx = int(annotation_path[1:-4])
                    timestamp = (1.0 / self.kaist_fps) * annotation_idx

                    annotations = self._read_annotations(
                        set_dir, vid_dir, annotation_path
                    )
                    for object_annotation in annotations:
                        dataset_annotations.append(
                            np.array(
                                (set_idx, vid_idx, timestamp, *object_annotation),
                                dtype=float,
                            )
                        )
        return pd.DataFrame(
            dataset_annotations,
            columns=[
                "set",
                "vid",
                "timestamp",
                *[field.name for field in dataclasses.fields(KaistPedObject)],
            ],
        )

    def _extract_objects(self, image_dir, annotations):
        previous_timestamp = 0.0
        for image_name in sorted(os.listdir(image_dir)):
            image_idx = int(image_name.replace(".png", ""))
            timestamp = self.target_frame_time * image_idx

            image = imageio.imread(f"{image_dir}/{image_name}")

            image_annotations = annotations[
                (annotations["timestamp"] >= previous_timestamp)
                & (annotations["timestamp"] < timestamp)
            ]
            for annotation in image_annotations.itertuples():
                annotation = annotation._asdict()
                x, y, width, height = self._make_square(
                    annotation["bbox_lefttop_x"],
                    annotation["bbox_lefttop_y"],
                    annotation["bbox_width"],
                    annotation["bbox_height"],
                    512,
                    640,
                )
                center_x = annotation["bbox_lefttop_x"] + (annotation["bbox_width"] / 2)
                center_y = annotation["bbox_lefttop_y"] + (
                    annotation["bbox_height"] / 2
                )

                yield self._standardise_shape(
                    image[y : y + height, x : x + width], 32
                ), annotation["label"], (int(center_x), int(center_y)), timestamp

            previous_timestamp = timestamp

    def _preprocess(self):
        train_path = f"{self.path}/{self.squares_dir}/{self.train_name}"
        test_path = f"{self.path}/{self.squares_dir}/{self.test_name}"

        mkpath(train_path)
        mkpath(test_path)

        annotations = self._aggregate_annotations()

        self._preprocess_spectrum(annotations, train_path, test_path, self.infrared_dir)
        self._preprocess_spectrum(annotations, train_path, test_path, self.visible_dir)

    def _preprocess_spectrum(self, annotations, train_path, test_path, spectrum_dir):
        mkpath(f"{train_path}/{spectrum_dir}")
        mkpath(f"{test_path}/{spectrum_dir}")

        train_labels = []
        test_labels = []

        for set_dir in tqdm(os.listdir(f"{self.path}/{self.events_dir}")):
            set_idx = int(set_dir[-2:])
            for vid_dir in os.listdir(f"{self.path}/{self.events_dir}/{set_dir}"):
                vid_idx = int(vid_dir[-2:])
                spectrum_path = (
                    f"{self.path}/{self.events_dir}/{set_dir}/{vid_dir}/{spectrum_dir}"
                )
                sequence_centers = np.empty((0, 2), dtype=int)
                last_visited = []

                for square, label, center, timestamp in self._extract_objects(
                    spectrum_path,
                    annotations[
                        (annotations["set"] == set_idx)
                        & (annotations["vid"] == vid_idx)
                    ],
                ):
                    if square.std() < self.std_thr:
                        continue

                    center = np.array(center)
                    distances = np.abs(sequence_centers - center).sum(axis=1)

                    if len(sequence_centers):
                        seq_idx = np.argmin(distances)
                    else:
                        seq_idx = None

                    if (
                        (seq_idx is not None)
                        and (distances[seq_idx] < self.center_eps)
                        and (abs(timestamp - last_visited[seq_idx]) < self.time_thr)
                    ):
                        sequence_centers[seq_idx] = np.array(center)
                        last_visited[seq_idx] = timestamp
                    else:
                        sequence_centers = np.append(
                            sequence_centers, center[np.newaxis, :], axis=0,
                        )
                        last_visited.append(timestamp)
                        seq_idx = len(sequence_centers) - 1

                    if self._is_train_set(set_dir):
                        imageio.imwrite(
                            f"{train_path}/{spectrum_dir}/{len(train_labels)}.png",
                            square,
                        )
                        train_labels.append([label, set_idx, vid_idx, seq_idx])
                    else:
                        imageio.imwrite(
                            f"{test_path}/{spectrum_dir}/{len(test_labels)}.png",
                            square,
                        )
                        test_labels.append([label, set_idx, vid_idx, seq_idx])

        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        np.savetxt(f"{train_path}/{spectrum_dir}.txt", train_labels, fmt="%d")
        np.savetxt(f"{test_path}/{spectrum_dir}.txt", test_labels, fmt="%d")

    def __len__(self):
        return len(
            os.listdir(f"{self.path}/{self.squares_dir}/{self.split}/{self.spectrum}")
        )

    def __getitem__(self, idx: int):
        if self.labels is None:
            self.labels = np.loadtxt(
                f"{self.path}/{self.squares_dir}/{self.split}/{self.spectrum}.txt",
                dtype=int,
            )

        if torch.is_tensor(idx) and idx.dim():
            return [self[jdx][0] for jdx in idx], [self[jdx][1] for jdx in idx]
        else:
            return (
                torchvision.io.read_image(
                    f"{self.path}/{self.squares_dir}/{self.split}/{self.spectrum}/{idx}.png"
                ),
                self.labels[idx][0],
            )


# %%
if __name__ == "__main__":
    dataset = KaistSquaresDataset()
    dataset._preprocess()
    print(len(dataset))
