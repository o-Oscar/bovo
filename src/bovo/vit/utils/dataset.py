from json import load

import numpy as np
import torch as th
import torch.nn as nn
from bovo.data.utils import BASE_PATH, read_db
from PIL import Image
from torch.utils.data import Dataset


def load_img(img_path, target_height, target_width):
    with Image.open(img_path) as image:
        image = image.resize((target_width, target_height))
        raw = np.asarray(image)
    raw = raw * 2 / 65535 - 1
    return raw


class VitDataset(Dataset):
    def __init__(
        self, train, img_size, train_fraction=0.9, transform=None, random=True
    ):
        self.img_size = img_size
        self.transform = transform
        self.random = random

        all_query_list = read_db(split_frac=train_fraction)
        self.query_list = all_query_list[0 if train else 1]

        # import matplotlib.pyplot as plt

        # plt.plot([x.po_bc for x in all_query_list[0]])
        # print("mean :", np.mean([x.po_bc > 17 for x in all_query_list[0]]))
        # plt.plot(
        #     list(
        #         range(
        #             len(all_query_list[0]),
        #             len(all_query_list[0]) + len(all_query_list[1]),
        #         )
        #     ),
        #     [x.po_bc for x in all_query_list[1]],
        # )
        # print("mean :", np.mean([x.po_bc > 17 for x in all_query_list[1]]))

        # plt.plot([0, len(all_query_list[0]) + len(all_query_list[1])], [17, 17], "--k")

        # plt.savefig("results/imgs/po_bc_list.png")
        # exit()

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
        image = load_img(img_path, self.img_size, self.img_size // 2)
        canvas = np.zeros((self.img_size, self.img_size))
        if self.random:
            dx = np.random.randint(self.img_size - image.shape[1])
        else:
            dx = (self.img_size - image.shape[1]) // 2
        canvas[:, dx : dx + image.shape[1]] = image
        canvas = np.expand_dims(canvas, axis=(0))
        source = th.Tensor(canvas)

        po_bc = self.query_list[idx].po_bc

        # sample = {"source": source, "po_bc": po_bc}
        sample = (source, po_bc)

        # import matplotlib.pyplot as plt

        # plt.imshow(canvas / 2 + 0.5)
        # plt.savefig("results/imgs/test.png")
        # print("imgsaved")
        # exit()
        # if self.transform:
        #     sample = self.transform(sample)

        return sample
