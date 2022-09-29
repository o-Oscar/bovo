import matplotlib.pyplot as plt
import numpy as np

import bovo

# trainset = bovo.data.get_train_set()
trainset = bovo.data.get_full_set()

trainset.sort_by_po_bc()


def imshow(img):
    plt.imshow(
        img,
        cmap="Greys_r",
        extent=(
            0,
            img.shape[1],
            0,
            img.shape[0],
        ),
    )
    plt.savefig("results/imgs/test.png", dpi=300)


r = (0, 100, 10)

lows = [trainset[i] for i in range(*r)]
lows_po_bc = [x["po_bc"] for x in lows]
lows = [x["image"] for x in lows]

highs = [trainset[len(trainset) - i - 1] for i in range(*r)][::-1]
highs_po_bc = [x["po_bc"] for x in highs]
highs = [x["image"] for x in highs]

lows = np.concatenate(lows, axis=1)
highs = np.concatenate(highs, axis=1)

imshow(np.concatenate([lows, highs], axis=0))
print(lows_po_bc)
print(highs_po_bc)
