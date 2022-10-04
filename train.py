import matplotlib.pyplot as plt
import numpy as np
import torch as th

import bovo.data.warp
import bovo.wrap.network
import wandb

USE_WANDB = True

# wandb.config = {
#     "layers": 1,
#     "first_depth": 4,
# }  # {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
print(device)

batch_size = 4

train_set = bovo.data.warp.WarpDataset(True, 0.9)
# train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
test_set = bovo.data.warp.WarpDataset(False, 0.9)
# test_loader = DataLoader(test_set, batch_size=10, shuffle=False)


def plot_network(save_id):
    batch = [train_set[100] for i in range(10)]

    I = th.concat([b["source"] for b in batch], dim=0).to(device)
    Ip = th.concat([b["warped"] for b in batch], dim=0).to(device)
    W = th.concat([b["warp"] for b in batch], dim=0).to(device)
    # J = th.concat([b["random"] for b in batch], dim=0).to(device)

    W_pred = net(I, Ip)

    I_np = np.concatenate(I.detach().cpu().numpy(), axis=2)[0]
    Ip_np = np.concatenate(Ip.detach().cpu().numpy(), axis=2)[0]
    to_plot = np.concatenate((I_np, Ip_np), axis=0)
    plt.imshow(to_plot, cmap="Greys_r")
    plt.savefig(f"results/imgs/tooth_{save_id}_img.png")

    W_np = np.concatenate(W.detach().cpu().numpy(), axis=1)
    W_pred_np = np.concatenate(W_pred.detach().cpu().numpy(), axis=1)
    to_plot = np.concatenate((W_np, W_pred_np), axis=0)
    ones = np.ones(to_plot.shape[:2] + (1,)) * 0.0
    to_plot = np.concatenate([to_plot, ones], axis=2) * 0.5 + 0.5
    plt.imshow(to_plot)
    plt.savefig(f"results/imgs/tooth_{save_id}_disp.png")


# net = bovo.wrap.network.FlowPredictionModule(64, 32, 1, n_layers=2, first_depth=32).to(
#     device
# )
# plot_network(0)
# exit()

configs = []
for n_layers in range(2, 4):
    for first_depth in [32, 64]:
        configs.append(
            {
                "n_layers": n_layers,
                "first_depth": first_depth,
            }
        )


for config in configs:

    model_name = "l{}_d{}".format(config["n_layers"], config["first_depth"])

    print("running new config :")
    print(config)

    net = bovo.wrap.network.FlowPredictionModule(64, 32, 1, **config).to(device)
    net.identity_wrap = net.identity_wrap.to(device)

    optimizer = th.optim.Adam(net.parameters(), lr=1e-4)

    plot_network(0)
    # exit()

    if USE_WANDB:
        wandb.init(
            project="bovo_warp",
            config=config,
        )

    lamb = 1

    for epoch in range(300):

        train_warp_loss = 0
        train_consistency_loss = 0
        batch_size = 10
        batch_nb = len(train_set) // batch_size
        all_idx = np.random.permutation(len(train_set))

        net.train()
        for batch_id in range(batch_nb):
            optimizer.zero_grad()

            batch = [
                train_set[all_idx[batch_id * batch_size + i]] for i in range(batch_size)
            ]

            I = th.concat([b["source"] for b in batch], dim=0).to(device)
            Ip = th.concat([b["warped"] for b in batch], dim=0).to(device)
            J = th.concat([b["random"] for b in batch], dim=0).to(device)
            W = th.concat([b["warp"] for b in batch], dim=0).to(device)

            warp_loss, consistency_loss = net.warp_losses(I, Ip, J, W)
            loss = warp_loss + consistency_loss * lamb
            loss.backward()
            optimizer.step()
            train_warp_loss += warp_loss.item() / batch_nb
            train_consistency_loss += consistency_loss.item() / batch_nb

        lamb = train_warp_loss / train_consistency_loss

        test_warp_loss = 0
        test_consistency_loss = 0
        net.eval()
        for batch_id in range(1):

            batch = [test_set[i] for i in range(batch_size)]

            I = th.concat([b["source"] for b in batch], dim=0).to(device)
            Ip = th.concat([b["warped"] for b in batch], dim=0).to(device)
            J = th.concat([b["random"] for b in batch], dim=0).to(device)
            W = th.concat([b["warp"] for b in batch], dim=0).to(device)

            warp_loss, consistency_loss = net.warp_losses(I, Ip, J, W)
            test_loss = 0
            test_warp_loss += warp_loss.item()
            test_consistency_loss += consistency_loss.item()

        to_save = {
            "train_warp_loss": train_warp_loss,
            "train_consistency_loss": train_consistency_loss,
            "test_warp_loss": test_warp_loss,
            "test_consistency_loss": test_consistency_loss,
        }
        if USE_WANDB:
            wandb.log(to_save)
        print(to_save)

        th.save(net.state_dict(), "results/models/" + model_name)

    plot_network(100)
    if USE_WANDB:
        wandb.finish()

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
