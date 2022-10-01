from json import load

import numpy as np
import torch as th
import torch.nn as nn
from bovo.data.utils import BASE_PATH, read_db
from PIL import Image
from torch.utils.data import Dataset


def load_img(img_path):
    target_width = 110
    target_height = 300
    margin = 100
    full_shape = [
        target_height + 2 * margin,
        target_width + 2 * margin,
    ]

    with Image.open(img_path) as image:
        # TODO : add random rotation to image
        raw = np.asarray(image)

    # TODO : add random color change
    raw = raw * 2 / 65535 - 1
    full = np.zeros(full_shape)
    dx = (full.shape[0] - raw.shape[0]) // 2
    dy = (full.shape[1] - raw.shape[1]) // 2
    # TODO : add random position drift
    full[dx : dx + raw.shape[0], dy : dy + raw.shape[1]] = raw

    end_image = full[
        margin:-margin,
        margin:-margin,
    ]
    return end_image


def get_random_warp(height, width):

    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    xx, yy = np.meshgrid(x, y)
    scale = np.random.uniform(0.5, 1)
    grid = np.expand_dims(np.stack((xx, yy), axis=2), axis=0) * scale
    grid = th.Tensor(grid)
    return grid


def get_identity_wrap(height, width):

    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    xx, yy = np.meshgrid(x, y)
    grid = np.expand_dims(np.stack((xx, yy), axis=2), axis=0)
    grid = th.Tensor(grid)
    return grid


class WarpDataset(Dataset):
    def __init__(self):
        self.query_list = read_db()
        self.target_height = 64
        self.target_width = 32

        self.to_resolution = nn.UpsamplingBilinear2d(
            size=(self.target_height, self.target_width)
        )

        self.identity_wrap = get_identity_wrap(self.target_height, self.target_width)

    def __len__(self):
        return len(self.query_list)

    def sort_by_po_bc(self):
        self.query_list.sort(key=lambda x: x.po_bc)
        print([x.po_bc for x in self.query_list])

    def __getitem__(self, idx):
        img_path = BASE_PATH / self.query_list[idx].chemin_cs
        image = load_img(img_path)
        image = np.expand_dims(image, axis=(0, 1))
        source = th.Tensor(image)
        source = self.to_resolution(source)

        random_warp = get_random_warp(self.target_height, self.target_width)
        warped = nn.functional.grid_sample(source, random_warp, align_corners=True)
        local_warp = random_warp - self.identity_wrap

        sample = {"source": source, "warped": warped, "warp": local_warp}
        return sample
