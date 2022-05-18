# %%
import torch
import torchvision

import numpy as np
import pandas as pd

# %%
from Datasets.KAIST_event import KaistSquaresDataset

# %%
class KaistSequencesDataset(KaistSquaresDataset):
    def __init__(
        self,
        path: str = "/datasets/KAIST_rgbt-ped-detection/data/kaist-rgbt",
        split: str = "train",
        spectrum: str = "visible",
        window_size: int = 3,
    ) -> None:
        super().__init__(path=path, split=split, spectrum=spectrum)
        self.window_size = window_size

        raw_labels = pd.DataFrame(
            np.loadtxt(
                f"{self.path}/{self.squares_dir}/{self.split}/{self.spectrum}.txt",
                dtype=int,
            ),
            columns=["label", "set_idx", "vid_idx", "seq_idx"],
        )
        raw_labels["img_idx"] = raw_labels.index
        filter_labels = dict(
            filter(
                lambda item: len(item[1]) > window_size,
                raw_labels.groupby(
                    ["label", "set_idx", "vid_idx", "seq_idx"], sort=False
                ).groups.items(),
            )
        )
        windowed_labels = dict(
            map(
                lambda item: (
                    item[0],
                    [
                        window.to_list()
                        for window in item[1].to_series().rolling(window_size)
                        if len(window) == window_size
                    ],
                ),
                filter_labels.items(),
            )
        )
        self.labels = [
            (label, window)
            for label, windows in windowed_labels.items()
            for window in windows
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx) and idx.dim():
            return (
                torch.stack([self[jdx][0] for jdx in idx]).squeeze(),
                [self[jdx][1] for jdx in idx],
            )
        else:
            return (
                torch.stack(
                    [
                        torchvision.io.read_image(
                            f"{self.path}/{self.squares_dir}/{self.split}/{self.spectrum}/{jdx}.png"
                        )
                        for jdx in self.labels[idx][-1]
                    ]
                ).squeeze(),
                self.labels[idx][0][0],
            )


# %%
if __name__ == "__main__":
    dataset = KaistSequencesDataset(
        "/datasets/KAIST_rgbt-ped-detection/data/kaist-rgbt"
    )
    print(dataset[42])
    print(len(dataset))
