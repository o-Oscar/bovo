import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import bovo.data.warp
import bovo.wrap.network

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

batch_size = 4

trainset = bovo.data.warp.WarpDataset()

net = bovo.wrap.network.FlowPredictionModule(64, 32, 1).to(device)

optimizer = th.optim.Adam(net.parameters(), lr=0.0001)


def plot_network(save_id):
    batch = [trainset[100] for i in range(10)]

    I = th.concat([b["source"] for b in batch], dim=0).to(device)
    J = th.concat([b["warped"] for b in batch], dim=0).to(device)
    W = th.concat([b["warp"] for b in batch], dim=0).to(device)
    W_pred = net(I, J)

    I_np = np.concatenate(I.detach().cpu().numpy(), axis=2)[0]
    J_np = np.concatenate(J.detach().cpu().numpy(), axis=2)[0]
    to_plot = np.concatenate((I_np, J_np), axis=0)
    plt.imshow(to_plot, cmap="Greys_r")
    plt.savefig(f"results/imgs/tooth_{save_id}_img.png")

    W_np = np.concatenate(W.detach().cpu().numpy(), axis=1)
    W_pred_np = np.concatenate(W_pred.detach().cpu().numpy(), axis=1)
    to_plot = np.concatenate((W_np, W_pred_np), axis=0)
    ones = np.ones(to_plot.shape[:2] + (1,)) * 0.0
    to_plot = np.concatenate([to_plot, ones], axis=2) * 0.5 + 0.5
    plt.imshow(to_plot)
    plt.savefig(f"results/imgs/tooth_{save_id}_disp.png")


plot_network(0)
# exit()

for epoch in range(100):

    total_loss = 0
    batch_nb = 10
    for batch_b in range(batch_nb):
        optimizer.zero_grad()

        batch = [trainset[i + 10 * batch_nb] for i in range(10)]

        I = th.concat([b["source"] for b in batch], dim=0).to(device)
        J = th.concat([b["warped"] for b in batch], dim=0).to(device)
        W = th.concat([b["warp"] for b in batch], dim=0).to(device)

        loss = net.loss(I, J, W)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(loss.item())

plot_network(100)

# W_pred = net(I, J)
# print(W_pred.shape)

# I_np = np.concatenate(I.detach().cpu().numpy(), axis=2)[0]
# J_np = np.concatenate(J.detach().cpu().numpy(), axis=2)[0]

# to_plot = np.concatenate((I_np, J_np), axis=0)
# plt.imshow(to_plot, cmap="Greys_r")
# plt.savefig("results/imgs/test.png")

# # gotta find a way to plot those : add a bunch of .5 for the b of rgb
# W_np = np.concatenate(W.detach().cpu().numpy(), axis=1)[0]
# W_pred_np = np.concatenate(W_pred.detach().cpu().numpy(), axis=1)[0]
