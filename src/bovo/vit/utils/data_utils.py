import logging

import torch
from bovo.vit.utils.dataset import VitDataset
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (args.img_size, args.img_size), scale=(0.05, 1.0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = (
            datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform_test
            )
            if args.local_rank in [-1, 0]
            else None
        )

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = (
            datasets.CIFAR100(
                root="./data", train=False, download=True, transform=transform_test
            )
            if args.local_rank in [-1, 0]
            else None
        )

    else:
        trainset = VitDataset(train=True, img_size=args.img_size)
        testset = VitDataset(train=False, img_size=args.img_size)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = (
        RandomSampler(trainset)
        if args.local_rank == -1
        else DistributedSampler(trainset)
    )
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            testset,
            sampler=test_sampler,
            batch_size=args.eval_batch_size,
            num_workers=2,
            pin_memory=True,
        )
        if testset is not None
        else None
    )

    return train_loader, test_loader
