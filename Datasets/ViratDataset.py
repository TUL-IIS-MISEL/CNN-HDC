import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from tqdm import tqdm

import torch
import torchvision
from tqdm import tqdm
import imageio
import itertools


class ViratObjectType(Enum):
    UNKNOWN = 0
    PERSON = 1
    CAR = 2
    VEHICLES = 3
    OBJECT = 4
    BIKE = 5


@dataclass
class ViratObject:
    object_id: int
    object_duration: int
    current_frame: int
    bbox_lefttop_x: int
    bbox_lefttop_y: int
    bbox_width: int
    bbox_height: int
    object_type: ViratObjectType

    @staticmethod
    def parse(text: str):
        chunks = text.strip().split(" ")
        if len(chunks) != 8:
            raise ValueError("Invalid input")
        return ViratObject(
            int(chunks[0]),
            int(chunks[1]),
            int(chunks[2]),
            int(chunks[3]),
            int(chunks[4]),
            int(chunks[5]),
            int(chunks[6]),
            ViratObjectType(int(chunks[7])),
        )

    @staticmethod
    def parse_file(infile):
        for line in infile:
            yield ViratObject.parse(line)


class ViratDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path="/datasets/VIRAT/Public Dataset/VIRAT Video Dataset Release 2.0/VIRAT Ground Dataset",
        *,
        initialize=True,
    ) -> None:
        super().__init__()
        self.path = path
        self.annotations_dir = "annotations"
        self.videos_dir = "videos_original"
        self.event_videos_dir = "videos_event_grey"
        self.names_path = "docs/list_release2.0.txt"
        self.data_infix = "viratdata"
        self.objects_suffix = "objects.txt"
        self.data: Dict[str, List[ViratObject]] = dict()

        if initialize:
            self._find_frames()

    def _find_frames(self):
        with open(f"{self.path}/{self.names_path}") as listfile:
            for name in tqdm(listfile):
                event_dir_path = f"{self.path}/{self.event_videos_dir}/{name.strip()}/"
                annotaion_file_path = f"{self.path}/{self.annotations_dir}/{name.strip()}.{self.data_infix}.{self.objects_suffix}"
                try:
                    # skip the first, malformed frame
                    vframes = sorted(os.listdir(event_dir_path))[1:]
                    with open(annotaion_file_path) as annotationfile:
                        annotations = list(ViratObject.parse_file(annotationfile))
                except FileNotFoundError:
                    # go to the next file
                    continue
                i2a = dict()
                for a in annotations:
                    if not a.current_frame in i2a:
                        i2a[a.current_frame] = []
                    i2a[a.current_frame].append(a)
                for frame in vframes:
                    idx = int(frame.split("_")[-1].split(".")[0])
                    self.data[f"{name.strip()}/{frame}"] = i2a.get(idx, [])
                # break

    def _preprocess_dataset_imageio(
        self, frame_step=4, transform=None,
    ):
        """
        Should be run only once to generate the event frames
        """
        with open(f"{self.path}/{self.names_path}") as listfile:
            for name in listfile:
                event_dir_path = f"{self.path}/{self.event_videos_dir}/{name.strip()}/"
                video_path = f"{self.path}/{self.videos_dir}/{name.strip()}.mp4"
                print(video_path)
                vid = imageio.get_reader(video_path, "ffmpeg")
                os.makedirs(event_dir_path, exist_ok=True)
                previous_frame = None
                idx = 0
                for vframe in tqdm(itertools.islice(vid, 0, None, frame_step)):
                    vframe = torchvision.transforms.ToTensor()(vframe)
                    if transform is not None:
                        vframe = transform(vframe)
                    if previous_frame is not None:
                        diff = vframe - previous_frame
                        diff /= 2.0
                        diff += 0.5
                        imageio.imwrite(
                            f"{event_dir_path}/img_{idx:06d}.png",
                            torchvision.transforms.ToPILImage()(diff),
                            format="png",
                            compress_level=0,
                        )
                    previous_frame = vframe
                    idx += frame_step
                # break

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = sorted(self.data.keys())[idx]
        if type(file) == str:
            return (
                torchvision.transforms.ToTensor()(
                    imageio.imread(f"{self.path}/{self.event_videos_dir}/{file}")
                ),
                self.data[file],
            )
        else:
            return (
                [
                    torchvision.transforms.ToTensor()(
                        imageio.imread(f"{self.path}/{self.event_videos_dir}/{f}")
                    )
                    for f in file
                ],
                [self.data[f] for f in file],
            )


if __name__ == "__main__":
    dataset = ViratDataset(initialize=False)

    preprocess_imageio = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((1080 // 4, 1920 // 4)),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]
    )
    dataset._preprocess_dataset_imageio(frame_step=4, transform=preprocess_imageio)
    dataset = ViratDataset()
