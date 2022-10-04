from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn
from PIL import Image

import bovo

res = bovo.data.read_db()

x: bovo.data.QueryResult


# base_path = Path("data/base_de_donnees_juin_2021/")
# all_sizes = []
# for x in res:
#     # print(x.chemin_cs)
#     file_path = base_path / x.chemin_cs
#     img = Image.open(file_path)
#     all_sizes.append(img.size)


# all_sizes = np.array(all_sizes)
# # plt.plot(all_sizes.T[0], all_sizes.T[1], ".b")
# # plt.savefig("results/imgs/test.png")

# db = pandas.DataFrame(data=res)
# db["img_width"] = all_sizes.T[0]
# db["img_height"] = all_sizes.T[1]

# seaborn.scatterplot(data=db, x="img_width", y="img_height", hue="po_bc")
# plt.savefig("results/imgs/test.png")
# print(db)
# exit()


# print(np.argmax(all_sizes.T[1]))
# print(res[1070])


# plt.plot(all_sizes)


all_paths = set()
for x in res:
    all_paths.add(
        Path("data/base_de_donnees_juin_2021/" + "/".join(Path(x.chemin_cs).parts[:2]))
    )

for path in all_paths:
    # print(path)
    bovo.gcp.download(path)
