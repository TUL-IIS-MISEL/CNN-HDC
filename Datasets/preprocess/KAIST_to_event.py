# Based on https://github.com/uzh-rpg/rpg_vid2e/blob/master/esim_py/README.md
# and https://github.com/uzh-rpg/rpg_vid2e/blob/master/esim_torch/generate_events.py
# %%
VID2E_DIR = "/home/pluczak/rpg_vid2e"

import sys
import os
from datetime import datetime
from multiprocessing import Pool

from distutils.dir_util import mkpath

import numpy as np
import imageio
from tqdm import tqdm

# %%
sys.path.append(VID2E_DIR)

import esim_py

# %%
# EVENT CONFIG
THR_CONTRAST_POSITIVE = 0.2  # contrast theshold for positive events
THR_CONTRAST_NEGATIVE = 0.2
# minimum waiting period (in sec) before a pixel can trigger a new event
REFRACTORY_PERIOD = 0
# epsilon that is used to numerical stability within the logarithm
EPSILON = 1e-3
USE_LOG_INTENSTY = True

# time for each accumulated frame
FRAME_TIME = 4.0 / 30.0  # in seconds (match VIRAT parameters)

# %%
KAIST_DATA_DIR = "/data/HDD_4_B/MISEL-datasets/KAIST_rgbt-ped-detection/data/kaist-rgbt"

# %%
def is_valid_dir(subdirs, files):
    return len(subdirs) == 2 and len(files) == 1 and "timestamps.txt" in files


# %%
def events_to_frames(
    events, output_dir: str, frame_time: float = FRAME_TIME, quiet: bool = False
):
    idx = 0
    t0 = 0
    frame = np.zeros((512, 640), dtype=int)
    for x, y, t, p in tqdm(events, disable=quiet):
        x = int(x)
        y = int(y)
        p = int(p)

        if t0 == 0:
            t0 = t
        if t - t0 > frame_time:
            if np.abs(frame).sum() > 0:  # ignore empty frames
                image = frame.astype(float)
                image = (image - image.min()) / (image.max() - image.min())
                image = (image * 255).astype("uint8")
                imageio.imwrite(f"{output_dir}/{idx:04d}.png", image)
            frame = np.zeros((512, 640), dtype=int)
            idx += 1
            t0 = t
        else:
            frame[y, x] += p


def process_dir(input_dir: str, output_dir: str, quiet: bool = False):
    esim = esim_py.EventSimulator(
        THR_CONTRAST_POSITIVE,
        THR_CONTRAST_NEGATIVE,
        REFRACTORY_PERIOD,
        EPSILON,
        USE_LOG_INTENSTY,
    )

    mkpath(output_dir)
    if len(os.listdir(output_dir)) > 0:
        if not quiet:
            print(f"{datetime.utcnow()}\t{output_dir} Not empty")
        return -1

    if not quiet:
        print(f"{datetime.utcnow()}\t{input_dir}")
    events_from_images = esim.generateFromFolder(
        input_dir, f"{input_dir}_timestamps.txt"
    )
    if not quiet:
        print(f"{datetime.utcnow()}\t{input_dir}")

    if not quiet:
        print(f"{datetime.utcnow()}\t{output_dir}")
    events_to_frames(events_from_images, output_dir, quiet=True)
    if not quiet:
        print(f"{datetime.utcnow()}\t{output_dir}")
    return 0


if __name__ == "__main__":
    quiet = False
    with Pool(processes=16) as pool:
        results = []
        print(f"{KAIST_DATA_DIR}/upsampled_images")
        for set_dir in os.listdir(f"{KAIST_DATA_DIR}/upsampled_images"):
            print(f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}")
            for vid_dir in os.listdir(f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}"):
                results.append(
                    pool.apply_async(
                        process_dir,
                        (
                            f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}/{vid_dir}/visible",
                            f"{KAIST_DATA_DIR}/events/{set_dir}/{vid_dir}/visible",
                            quiet,
                        ),
                    )
                )
                results.append(
                    pool.apply_async(
                        process_dir,
                        (
                            f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}/{vid_dir}/lwir",
                            f"{KAIST_DATA_DIR}/events/{set_dir}/{vid_dir}/lwir",
                            quiet,
                        ),
                    )
                )
        pool.close()
        pool.join()
        for result in tqdm(results):
            print(result.get())

    # print(f"{KAIST_DATA_DIR}/upsampled_images")
    # for set_dir in os.listdir(f"{KAIST_DATA_DIR}/upsampled_images"):
    #     print(f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}")
    #     for vid_dir in os.listdir(f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}"):
    #         process_dir(
    #             f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}/{vid_dir}/visible",
    #             f"{KAIST_DATA_DIR}/events/{set_dir}/{vid_dir}/visible",
    #         )

    #         process_dir(
    #             f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}/{vid_dir}/lwir",
    #             f"{KAIST_DATA_DIR}/events/{set_dir}/{vid_dir}/lwir",
    #         )

