# %%
import os

from enum import Enum
from dataclasses import dataclass

# %%
import torch
import imageio
import h5py

import numpy as np

from tqdm import tqdm

# %%
class KaistPedObjectType(Enum):
    UNKNOWN = 0
    PERSON = 1
    PEOPLE = 2
    CYCLIST = 3

    @staticmethod
    def parse(name: str):
        name_map = {
            "person?": KaistPedObjectType.UNKNOWN,
            "person": KaistPedObjectType.PERSON,
            "people": KaistPedObjectType.PEOPLE,
            "cyclist": KaistPedObjectType.CYCLIST,
        }
        if name not in name_map.keys():
            raise ValueError
        return name_map[name]


# %%
@dataclass
class KaistPedObject:
    label: KaistPedObjectType
    bbox_lefttop_x: int
    bbox_lefttop_y: int
    bbox_width: int
    bbox_height: int
    occlusion: int
    bbox_visible_lefttop_x: int
    bbox_visible_lefttop_y: int
    bbox_visible_width: int
    bbox_visible_height: int
    ignore: int
    angle: int

    @staticmethod
    def parse(text: str):
        chunks = text.strip().split(" ")
        if text[0] == "%":
            return None
        if len(chunks) != 12:
            raise ValueError("Invalid input")

        return KaistPedObject(
            KaistPedObjectType.parse(chunks[0]),
            int(chunks[1]),
            int(chunks[2]),
            int(chunks[3]),
            int(chunks[4]),
            int(chunks[5]),
            int(chunks[6]),
            int(chunks[7]),
            int(chunks[8]),
            int(chunks[9]),
            int(chunks[10]),
            int(chunks[11]),
        )

    @staticmethod
    def parse_file(infile):
        for line in infile:
            yield KaistPedObject.parse(line)

    def to_numpy(self):
        return np.array(
            [
                self.label.value,
                self.bbox_lefttop_x,
                self.bbox_lefttop_y,
                self.bbox_width,
                self.bbox_height,
                self.occlusion,
                self.bbox_visible_lefttop_x,
                self.bbox_visible_lefttop_y,
                self.bbox_visible_width,
                self.bbox_visible_height,
                self.ignore,
                self.angle,
            ]
        )

    @staticmethod
    def from_numpy(vector):
        return KaistPedObject(
            KaistPedObjectType(vector[0]),
            int(vector[1]),
            int(vector[2]),
            int(vector[3]),
            int(vector[4]),
            int(vector[5]),
            int(vector[6]),
            int(vector[7]),
            int(vector[8]),
            int(vector[9]),
            int(vector[10]),
            int(vector[11]),
        )


# %%
class KaistPedDataset(torch.utils.data.Dataset):
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

        self.annotations_dir = "annotations"
        self.images_dir = "images"
        self.visible_dir = "visible"
        self.infrared_dir = "lwir"
        self.train_suffixes = {"00", "01", "02", "03", "04", "05"}
        self.test_suffixes = {"06", "07", "08", "09", "10", "11"}

        self.hdf5_name = "KAIST_PED"

    def _preprocess(
        self, dry_run: bool = True, count_train: int = 0, count_test: int = 0,
    ):
        if not dry_run:
            with h5py.File(f"{self.path}/{self.hdf5_name}.h5", "w") as file:
                visible_group = file.create_group("visible")
                infrared_group = file.create_group("infrared")
                annotation_group = file.create_group("annotations")
                vid_index_group = file.create_group("vid_idx")

                visible_group.create_dataset(
                    "train", (count_train, 512, 640, 3), h5py.h5t.STD_U8LE
                )
                visible_group.create_dataset(
                    "test", (count_test, 512, 640, 3), h5py.h5t.STD_U8LE
                )

                infrared_group.create_dataset(
                    "train", (count_train, 512, 640, 3), h5py.h5t.STD_U8LE
                )
                infrared_group.create_dataset(
                    "test", (count_test, 512, 640, 3), h5py.h5t.STD_U8LE
                )

                annotation_group.create_dataset(
                    "train", (count_train,), h5py.vlen_dtype(h5py.h5t.STD_U16LE)
                )
                annotation_group.create_dataset(
                    "test", (count_test,), h5py.vlen_dtype(h5py.h5t.STD_U16LE)
                )

                vid_index_group.create_dataset(
                    "train", (count_train, 2), h5py.h5t.STD_U8LE
                )
                vid_index_group.create_dataset(
                    "test", (count_test, 2), h5py.h5t.STD_U8LE
                )

        pbar = tqdm(total=count_train + count_test)

        count_train = 0
        count_test = 0

        for set_dir in os.listdir(f"{self.path}/{self.annotations_dir}"):
            for vid_dir in os.listdir(f"{self.path}/{self.annotations_dir}/{set_dir}"):
                for annotation_path in os.listdir(
                    f"{self.path}/{self.annotations_dir}/{set_dir}/{vid_dir}"
                ):
                    image_path = f"{self.path}/{self.images_dir}/{set_dir}/{vid_dir}"
                    file_name = annotation_path.replace(".txt", ".jpg")
                    visible_path = f"{image_path}/{self.visible_dir}/{file_name}"
                    infrared_path = f"{image_path}/{self.infrared_dir}/{file_name}"

                    subset = "train" if set_dir[-2:] in self.train_suffixes else "test"

                    if not dry_run:
                        visible = imageio.imread(visible_path)
                        infrared = imageio.imread(infrared_path)
                        annotations = self._read_annotations(
                            set_dir, vid_dir, annotation_path
                        )
                        with h5py.File(
                            f"{self.path}/{self.hdf5_name}.h5", "r+"
                        ) as file:
                            file[f"/visible/{subset}"][
                                count_train if subset == "train" else count_test
                            ] = visible
                            file[f"/infrared/{subset}"][
                                count_train if subset == "train" else count_test
                            ] = infrared
                            file[f"/annotations/{subset}"][
                                count_train if subset == "train" else count_test
                            ] = annotations.flatten()
                            file[f"/vid_idx/{subset}"][
                                count_train if subset == "train" else count_test
                            ] = np.array([int(set_dir[-2:]), int(vid_dir[-3:])])

                    if subset == "train":
                        count_train += 1
                    else:
                        count_test += 1
                    pbar.update(1)
        pbar.close()

        return count_train, count_test

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

    def __len__(self):
        datafile = h5py.File(f"{self.path}/{self.hdf5_name}.h5", "r")
        return len(datafile["visible"][self.split])

    def __getitem__(self, idx: int):
        datafile = h5py.File(f"{self.path}/{self.hdf5_name}.h5", "r")
        return (
            datafile["visible"][self.split][idx],
            datafile["infrared"][self.split][idx],
            datafile["annotations"][self.split][idx].reshape(-1, 12),
            datafile["vid_idx"][self.split][idx],
        )


# %%
if __name__ == "__main__":
    dataset = KaistPedDataset()
    (count_train, count_test,) = dataset._preprocess(dry_run=True)
    # it takes 1h to preallocate the file before processing actually starts
    dataset._preprocess(False, count_train, count_test)
