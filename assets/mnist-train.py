# Basic routines for trainning DL model
# Source: https://nextjournal.com/gkoehler/pytorch-mnist
import os
import torch
import tvault
import argparse
import torchvision
import numpy as np
import torch.optim as optim
import torch.distributed as dist

from module import resnet18
from torch.nn.parallel import DistributedDataParallel as DDP


# seeding
seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
np.random.seed(seed)

# configurations
batch_size = 4096
learning_rate = 1e-3
log_interval = 1


def train(model, train_epoch, train_loader, local_rank, criterion):
    model.train()
    loss_acc = 0
    for epoch in range(train_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(local_rank)
            target = target.to(local_rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss_acc += loss.item() / batch_size
            loss.backward()
            optimizer.step()
        if epoch % log_interval == 0:
            print(f"Train Epoch: {epoch} \tLoss: {loss_acc / len(train_loader)}")


def test(model, test_loader, local_rank, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(local_rank)
            target = target.to(local_rank)
            output = model(data)
            test_loss += criterion(output, target).item()  # size avg?
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    return 100.0 * correct / len(test_loader.dataset)


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--gpu_ids", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    return parser


def init_for_distributed(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", init_method="env://")
    if args.local_rank is not None:
        args.local_rank = local_rank
        print("Use GPU: {} for training".format(args.local_rank))
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MNIST arg parser", parents=[get_args_parser()])
    args = parser.parse_args()

    # DDP
    init_for_distributed(args)

    # Model
    train_dataset = torchvision.datasets.MNIST(
        "/MNIST/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )

    test_dataset = torchvision.datasets.MNIST(
        "/MNIST/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
    )

    for learning_rate in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
        model = resnet18(10)
        print(f"start training for lr {learning_rate}")

        model = model.to(args.local_rank)
        model = DDP(model, device_ids=[args.local_rank])
        criterion = torch.nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        train(model, 5, train_loader, args.local_rank, criterion)
        if args.local_rank == 0:
            acc = test(model, test_loader, args.local_rank, criterion)
        tags = {"language": "pytorch", "size": "0.5x", "learning_rate": learning_rate}
        tvault.log_all(model, tags=tags, result=acc.item(), optimizer=optimizer)
