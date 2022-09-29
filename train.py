import torch
import torchvision
import torchvision.transforms as transforms

import bovo

# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# )

batch_size = 4

trainset = bovo.data.get_train_set()
print(trainset[10])

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testset = bovo.data.get_test_set()
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

classes = ("dente", "edente")

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


# get some random training images
dataiter = iter(trainloader)
data = dataiter.next()

# show images
imshow(data["image"])
# print labels
# print(" ".join(f"{data["po_bc"][j]:5s}" for j in range(batch_size)))
