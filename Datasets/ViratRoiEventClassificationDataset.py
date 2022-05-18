import os

from tqdm import tqdm
import imageio
import numpy as np

from Datasets.ViratDataset import ViratDataset, ViratObjectType
from Datasets.ViratEventClassificationDataset import ViratEventClassificationDataset

import cv2 as cv


class ViratRoiEventClassificationDataset(ViratEventClassificationDataset):
    def __init__(self, path="/datasets/VIRAT_ROI") -> None:
        super().__init__(path=path)

    def _blobize_image(self, image):
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

        img = (np.abs(image - 0.5) * 255).astype(np.uint8)
        img = cv.filter2D(img, -1, (1.0 / kernel.sum()) * kernel)
        img = cv.filter2D(img, -1, (1.0 / kernel.sum()) * kernel)
        _, img = cv.threshold(img, 0.01 * 255, 255, cv.THRESH_BINARY)
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)
        img = cv.dilate(img, kernel, iterations=4)
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=3)
        img = cv.dilate(img, kernel, iterations=3)
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=1)
        return img

    def _generate_roi(self, blobized_image, image, ratio_thr=0.5):
        count, obj = cv.connectedComponents(blobized_image, connectivity=4)
        for i in range(1, count):
            obj_mask = obj == i
            x, y, width, height = self._roi_bounding_box(obj_mask)
            background = image.mean()
            extracted = (image * obj_mask)[y : y + height, x : x + width] - background
            extracted_abs = np.abs(extracted)
            extracted_range = np.ptp(extracted) / 2
            if (
                ((obj_mask.sum() / (width * height)) > ratio_thr)
                and (
                    (
                        (extracted_abs > (0.2 * extracted_range)).sum()
                        / (width * height)
                    )
                    > ratio_thr  # za ostre kryterium
                )
                and (width >= (blobized_image.shape[1] * 0.05))
                and (height >= (blobized_image.shape[0] * 0.05))
                # and abs((extracted < 0.0).sum() / (extracted > 0.0).sum() - 1) < 0.35
            ):
                yield obj_mask

    def _roi_bounding_box(self, roi_mask):
        x, y, width, height = cv.boundingRect(roi_mask.astype(np.uint8))
        return x, y, width, height

    def _check_collision(self, rect1, rect2):
        xl1, yt1, width1, height1 = rect1
        xc1 = xl1 + 0.5 * width1
        yc1 = yt1 + 0.5 * height1

        xl2, yt2, width2, height2 = rect2
        xc2 = xl2 + 0.5 * width2
        yc2 = yt2 + 0.5 * height2

        xthresh = 0.5 * (width1 + width2)
        ythresh = 0.5 * (height1 + height2)

        xdist = abs(xc1 - xc2)
        ydist = abs(yc1 - yc2)

        return (xdist < xthresh) and (ydist < ythresh), (xdist + ydist)

    def _preprocess_dataset(
        self,
        virat_dataset: ViratDataset,
        shrinkage: int = 4,
        side=100,
        ratio_thr=0.01,
        preview=False,
    ):
        os.makedirs(self.path, exist_ok=True)
        file_list = []
        for idx in tqdm(range(len(virat_dataset))):
            image, objects = virat_dataset[idx]
            img = np.transpose(image.numpy(), (1, 2, 0)).squeeze()
            if preview:
                view = img.copy()

            for rdx, roi in enumerate(
                self._generate_roi(self._blobize_image(img), img, ratio_thr)
            ):
                box = self._roi_bounding_box(roi)
                min_distance = np.inf
                candidate_label = ViratObjectType.UNKNOWN
                for obj in objects:
                    intersect, dist = self._check_collision(
                        box,
                        (
                            obj.bbox_lefttop_x // shrinkage,
                            obj.bbox_lefttop_y // shrinkage,
                            obj.bbox_width // shrinkage,
                            obj.bbox_height // shrinkage,
                        ),
                    )
                    if intersect and min_distance > dist:
                        min_distance = dist
                        candidate_label = obj.object_type

                if (
                    candidate_label != ViratObjectType.CAR
                    and candidate_label != ViratObjectType.PERSON
                ):
                    continue

                x, y, width, height = self._make_square(
                    *box, img.shape[1], img.shape[0],
                )
                square = img[y : y + height, x : x + width]
                square = cv.resize(square, (side, side))
                square *= 255
                square = square.astype(np.uint8)

                file_name = f"img_{idx:08d}_{rdx:04d}_{candidate_label.value:02d}.png"
                imageio.imwrite(
                    f"{self.path}/{file_name}", square, format="png", compress_level=0,
                )
                file_list.append(f"{file_name};{candidate_label.value:02d}\n")

                if preview:
                    cv.rectangle(
                        view,
                        (x, y),
                        (x + width, y + height,),
                        {ViratObjectType.CAR: 1, ViratObjectType.PERSON: 0,}.get(
                            candidate_label, 0.75
                        ),
                    )

            if preview:
                cv.imshow("", view)
                cv.setWindowTitle("", f"{idx}")
                cv.waitKey(1)

        with open(f"{self.path}/{self.list_file}", "w") as outfile:
            for line in file_list:
                outfile.write(line)


if __name__ == "__main__":
    import h5py

    source_dataset = ViratDataset(initialize=False)
    source_dataset.event_videos_dir = "videos_event_grey_VGA_8"
    source_dataset._find_frames()

    dataset = ViratRoiEventClassificationDataset(
        path="/datasets/VIRAT_ROI_VGA_32"
    )
    dataset._preprocess_dataset(
        source_dataset, preview=True, side=32, shrinkage=3, ratio_thr=0.5,
    )
    dataset._convert_to_hdf5()

    from collections import Counter

    dataset = ViratRoiEventClassificationDataset(
        path="/datasets/VIRAT_ROI_VGA_32"
    )
    print(Counter(h5py.File(f"{dataset.path}.h5", "r")["labels"]))
