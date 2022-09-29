import matplotlib.pyplot as plt
import numpy as np


def imshow(imgs):
    img = np.concatenate([img.numpy() for img in imgs], axis=1)
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
    plt.savefig("results/imgs/test.png")
