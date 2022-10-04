import matplotlib.pyplot as plt
import numpy as np
import torch as th

import bovo.data.warp

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
print(device)

batch_size = 4

train_set = bovo.data.warp.WarpDataset(True, 0.9)
# train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
test_set = bovo.data.warp.WarpDataset(False, 0.9)
# test_loader = DataLoader(test_set, batch_size=10, shuffle=False)


def plot_db(save_id):
    batch = [train_set[100] for i in range(10)]

    I = th.concat([b["source"] for b in batch], dim=0).to(device)
    J = th.concat([b["warped"] for b in batch], dim=0).to(device)
    W = th.concat([b["warp"] for b in batch], dim=0).to(device)

    I_np = np.concatenate(I.detach().cpu().numpy(), axis=2)[0]
    J_np = np.concatenate(J.detach().cpu().numpy(), axis=2)[0]
    to_plot = np.concatenate((I_np, J_np), axis=0)
    plt.imshow(to_plot, cmap="Greys_r")
    plt.savefig(f"results/imgs/tooth_{save_id}_img.png")

    W_np = np.concatenate(W.detach().cpu().numpy(), axis=1)
    to_plot = np.concatenate((W_np,), axis=0)
    ones = np.ones(to_plot.shape[:2] + (1,)) * 0.0
    to_plot = np.concatenate([to_plot, ones], axis=2) * 0.5 + 0.5
    plt.imshow(to_plot)
    plt.savefig(f"results/imgs/tooth_{save_id}_disp.png")


plot_db(1)
