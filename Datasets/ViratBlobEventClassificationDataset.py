import numpy as np

from Datasets.ViratDataset import ViratDataset
from Datasets.ViratRoiEventClassificationDataset import (
    ViratRoiEventClassificationDataset,
)

import cv2 as cv


class ViratBlobEventClassificationDataset(ViratRoiEventClassificationDataset):
    def __init__(self, path="/datasets/VIRAT_BLOB") -> None:
        super().__init__(path=path)

    def _blobize_image(self, image):
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

        img = (np.abs(image - 0.5) * 255).astype(np.uint8)
        img = cv.filter2D(img, -1, (1.0 / kernel.sum()) * kernel)
        img = cv.filter2D(img, -1, (1.0 / kernel.sum()) * kernel)
        _, img = cv.threshold(img, 0.01 * 255, 255, cv.THRESH_BINARY)
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=4)
        img = cv.erode(img, kernel, iterations=1)
        return img


if __name__ == "__main__":
    import h5py

    # import torchvision

    # source_dataset = ViratDataset()
    source_dataset = ViratDataset(initialize=False)
    source_dataset.event_videos_dir = "videos_event_grey_VGA_8"

    # preprocess_imageio = torchvision.transforms.Compose(
    #     [
    #         torchvision.transforms.Resize(
    #             (1080 // 3, 1920 // 3)
    #         ),  # down to 640 x 360 (close to VGA)
    #         torchvision.transforms.Grayscale(num_output_channels=1),
    #     ]
    # )
    # source_dataset._preprocess_dataset_imageio(
    #     frame_step=8, transform=preprocess_imageio
    # )  # 30 FPS / 8 ~= 3.75 FPS
    source_dataset._find_frames()

    dataset = ViratBlobEventClassificationDataset()
    # dataset._preprocess_dataset(source_dataset, preview=True, side=32, shrinkage=4)
    # dataset._preprocess_dataset(
    #     source_dataset, preview=True, side=32, shrinkage=3, ratio_thr=0.5,
    # )  # 0.45 still allows for "empty noise", 0.5 passes only very crisp images
    # dataset._convert_to_hdf5()

    from collections import Counter

    dataset = ViratBlobEventClassificationDataset()
    print(Counter(h5py.File(f"{dataset.path}.h5", "r")["labels"]))
