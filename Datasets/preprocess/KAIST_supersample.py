# Based on https://github.com/uzh-rpg/rpg_vid2e/blob/master/esim_py/README.md
# and https://github.com/uzh-rpg/rpg_vid2e/blob/master/esim_torch/generate_events.py
# %%
VID2E_DIR = "/home/pluczak/rpg_vid2e"

import sys
import os
from distutils.file_util import move_file

sys.path.append(VID2E_DIR)

# %%
import upsampling.utils.const

assert (
    upsampling.utils.const.imgs_dirname == ""
), "This has to be replaced manually for the code to work"

from upsampling.utils import Upsampler

# %%
KAIST_DATA_DIR = "/data/HDD_4_B/MISEL-datasets/KAIST_rgbt-ped-detection/data/kaist-rgbt"
KAIST_FPS = 20

# %%
def is_valid_dir(subdirs, files):
    return len(subdirs) == 2 and len(files) == 1 and "timestamps.txt" in files


# %%
if __name__ == "__main__":
    print(os.listdir(f"{KAIST_DATA_DIR}/annotations"))
    for set_dir in os.listdir(f"{KAIST_DATA_DIR}/annotations"):
        print(os.listdir(f"{KAIST_DATA_DIR}/annotations/{set_dir}"))
        for vid_dir in os.listdir(f"{KAIST_DATA_DIR}/annotations/{set_dir}"):
            with open(
                f"{KAIST_DATA_DIR}/images/{set_dir}/{vid_dir}/visible/fps.txt", "w",
            ) as fps_file:
                fps_file.write(str(KAIST_FPS))
            with open(
                f"{KAIST_DATA_DIR}/images/{set_dir}/{vid_dir}/lwir/fps.txt", "w"
            ) as fps_file:
                fps_file.write(str(KAIST_FPS))

            try:
                print(f"{KAIST_DATA_DIR}/images/{set_dir}/{vid_dir}/visible")
                upsampler = Upsampler(
                    input_dir=f"{KAIST_DATA_DIR}/images/{set_dir}/{vid_dir}/visible",
                    output_dir=f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}/{vid_dir}/visible",
                    device="cuda",
                )
                upsampler.upsample()
                move_file(
                    f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}/{vid_dir}/visible/timestamps.txt",
                    f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}/{vid_dir}/visible_timestamps.txt",
                )

                print(f"{KAIST_DATA_DIR}/images/{set_dir}/{vid_dir}/lwir")
                upsampler = Upsampler(
                    input_dir=f"{KAIST_DATA_DIR}/images/{set_dir}/{vid_dir}/lwir",
                    output_dir=f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}/{vid_dir}/lwir",
                    device="cuda",
                )
                upsampler.upsample()
                move_file(
                    f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}/{vid_dir}/lwir/timestamps.txt",
                    f"{KAIST_DATA_DIR}/upsampled_images/{set_dir}/{vid_dir}/lwir_timestamps.txt",
                )
            except AssertionError as err:
                print(err)
