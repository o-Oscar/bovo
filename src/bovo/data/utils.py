import dataclasses
import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

BASE_PATH = Path("data/base_de_donnees_juin_2021")
DB_PATH = BASE_PATH / "Coupes_sagittales_classification/database_CS_classif.db"

_connection = sqlite3.connect(DB_PATH)
_cursor = _connection.cursor()


@dataclasses.dataclass
class QueryResult:
    id_exam: int = 0
    id_cs: int = 0
    chemin_cs: str = 0
    po_bc: float = 0


def read_db():
    query = f""" select * from (select id_examen, id_cs, chemin_cs, po_bc from coupes_sagittales_classification where creux = 1 and dente = 1)"""
    _cursor.execute(query)
    return [QueryResult(*x) for x in _cursor.fetchall()]


def get_filtered_results():
    results = read_db()
    return [x for x in results if x.po_bc > -10]


def get_full_set():
    query_list = get_filtered_results()
    return ToothDataset(query_list)


def get_train_set():
    query_list = get_filtered_results()
    return ToothDataset(query_list[:500])


def get_test_set():
    query_list = get_filtered_results()
    return ToothDataset(query_list[50:])


class ToothDataset(Dataset):
    def __init__(self, query_list: list[QueryResult]):
        self.query_list = query_list
        self.target_width = 110
        self.target_height = 300
        self.margin = 100
        self.full_shape = [
            self.target_height + 2 * self.margin,
            self.target_width + 2 * self.margin,
        ]

    def __len__(self):
        return len(self.query_list)

    def sort_by_po_bc(self):
        self.query_list.sort(key=lambda x: x.po_bc)
        print([x.po_bc for x in self.query_list])

    def __getitem__(self, idx):
        img_path = BASE_PATH / self.query_list[idx].chemin_cs
        with Image.open(img_path) as image:
            # TODO : add random rotation to image
            image.resize((basewidth, hsize), Image.ANTIALIAS)

            raw = np.asarray(image)

        # TODO : add random color change
        raw = raw * 2 / 65535 - 1
        full = np.zeros(self.full_shape)
        dx = (full.shape[0] - raw.shape[0]) // 2
        dy = (full.shape[1] - raw.shape[1]) // 2
        # TODO : add random position drift
        full[dx : dx + raw.shape[0], dy : dy + raw.shape[1]] = raw

        end_image = full[
            self.margin : -self.margin,
            self.margin : -self.margin,
        ]

        sample = {"image": end_image, "po_bc": self.query_list[idx].po_bc}

        return sample
