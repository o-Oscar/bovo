from json import load

import numpy as np
import torch as th
import torch.nn as nn
from bovo.data.utils import BASE_PATH, read_db
from PIL import Image
from torch.utils.data import Dataset


def load_img(img_path):
    target_width = 200
    target_height = 600
    with Image.open(img_path) as image:
        image = image.resize((target_width, target_height))
        raw = np.asarray(image)
    raw = raw * 2 / 65535 - 1
    return raw


def get_random_warp(target_height, target_width, scale):

    height = 3
    width = 2
    s_x = scale / target_width
    s_y = scale / target_height

    dx = np.random.normal(size=(height, width)) * s_x
    dy = np.random.normal(size=(height, width)) * s_y
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    xx, yy = np.meshgrid(x, y)

    grid = np.expand_dims(np.stack((dx + xx, dy + yy), axis=0), axis=0)
    grid = th.permute(
        nn.functional.interpolate(
            th.Tensor(grid),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=True,
        ),
        dims=(0, 2, 3, 1),
    )
    return grid


def get_identity_wrap(height, width):

    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    xx, yy = np.meshgrid(x, y)
    grid = np.expand_dims(np.stack((xx, yy), axis=2), axis=0)
    grid = th.Tensor(grid)
    return grid


class WarpDataset(Dataset):
    def __init__(self, is_train_set, train_fraction):
        self.query_list = read_db()
        print(len(self.query_list))

        train_id = int(len(self.query_list) * (1 - train_fraction))
        if is_train_set:
            self.query_list = self.query_list[train_id:]
        else:
            self.query_list = self.query_list[:train_id]

        self.hires_height = 600
        self.hires_width = 300
        self.target_height = 64
        self.target_width = 32

        self.to_resolution = nn.UpsamplingBilinear2d(
            size=(self.target_height, self.target_width)
        )

        self.identity_wrap = get_identity_wrap(self.hires_height, self.hires_width)

    def __len__(self):
        return len(self.query_list)

    def sort_by_po_bc(self):
        self.query_list.sort(key=lambda x: x.po_bc)

    # def __getitem__(self, idx):
    #     img_path = BASE_PATH / self.query_list[idx].chemin_cs
    #     image = load_img(img_path)
    #     image = np.expand_dims(image, axis=(0, 1))
    #     source = th.Tensor(image)
    #     source = self.to_resolution(source)

    #     random_warp = get_random_warp(self.target_height, self.target_width)
    #     warped = nn.functional.grid_sample(
    #         source, random_warp, align_corners=True, padding_mode="border"
    #     )
    #     local_warp = random_warp - self.identity_wrap

    #     sample = {"source": source, "warped": warped, "warp": local_warp}
    #     return sample

    def __getitem__(self, idx):
        img_path = BASE_PATH / self.query_list[idx].chemin_cs
        image = load_img(img_path)
        image = np.expand_dims(image, axis=(0, 1))
        source = th.Tensor(image)

        random_warp = get_random_warp(self.hires_height, self.hires_width, 50)
        warped = nn.functional.grid_sample(
            source, random_warp, align_corners=True, padding_mode="border"
        )
        local_warp = random_warp - self.identity_wrap

        local_warp = th.permute(
            self.to_resolution(th.permute(local_warp, dims=(0, 3, 1, 2))),
            dims=(0, 2, 3, 1),
        )
        warped = self.to_resolution(warped)
        source = self.to_resolution(source)

        rand_idx = np.random.randint(len(self.query_list))
        img_path = BASE_PATH / self.query_list[rand_idx].chemin_cs
        image = load_img(img_path)
        image = np.expand_dims(image, axis=(0, 1))
        random = th.Tensor(image)
        random = self.to_resolution(random)

        sample = {
            "source": source,
            "warped": warped,
            "warp": local_warp,
            "random": random,
        }
        return sample
