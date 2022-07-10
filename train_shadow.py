from shadow.trainer import train
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.optim import Adam, AdamW
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import random
from easydict import EasyDict
import yaml
import wandb

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])

# conduct training
if not os.path.exists(CFG.save_path):
    os.makedirs(CFG.save_path)

# seed for future replication
np.random.seed(CFG.seed)
random.seed(CFG.seed)

# Load the CIFAR dataset
# CIFAR train is used for shadow model train & evaluation
# CIFAR test is used for target model train & evaluation
if CFG.num_classes == 10:
    DSET_CLASS = torchvision.datasets.CIFAR10
elif CFG.num_classes == 100:
    DSET_CLASS = torchvision.datasets.CIFAR100

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = DSET_CLASS(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=2
)

testset = DSET_CLASS(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=CFG.val_batch_size, shuffle=False, num_workers=2
)


# Define Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# random subset for shadow model train & validation from the CIFAR trainset
list_train_loader = []
list_eval_loader = []

for _ in range(CFG.num_shadow_models):
    train_indices = np.random.choice(len(trainset), CFG.shadow_train_size, replace=False)

    eval_indices = np.setdiff1d(np.arange(len(trainset)), train_indices)

    subset_train = torch.utils.data.Subset(trainset, train_indices)

    subset_eval = torch.utils.data.Subset(trainset, eval_indices)

    subset_train_loader = torch.utils.data.DataLoader(
        subset_train, batch_size=CFG.train_batch_size, shuffle=True, num_workers=2
    )

    subset_eval_loader = torch.utils.data.DataLoader(
        subset_eval, batch_size=CFG.val_batch_size, shuffle=False, num_workers=2
    )

    list_train_loader.append(subset_train_loader)
    list_eval_loader.append(subset_eval_loader)

# Train shadow model
for shadow_number, trainloader in enumerate(tqdm(list_train_loader)):
    shadow_model = resnet18(pretrained=False)
    shadow_model = shadow_model.to(device)

    run_name = f"resnet18_shadow_{shadow_number}"

    wandb.init(
        entity="cysec",
        project="membership_inference_attack",
        group=f"{shadow_model.__class__.__name__}_shadow",
        name=run_name,
    )

    optimizer = AdamW(
        shadow_model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )

    train(
        CFG,
        shadow_model,
        trainloader,
        list_eval_loader[shadow_number],
        optimizer,
        CFG.save_path,
        shadow_number,
        scheduler=None,
    )

    # Prevent OOM error
    shadow_model.cpu()
    del shadow_model
    del optimizer
    torch.cuda.empty_cache()
    wandb.finish()

# Train Target Model
target_model = resnet18(pretrained=False)
target_model = target_model.to(device)
optimizer = AdamW(target_model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)

target_train_indices = np.random.choice(len(testset), CFG.shadow_train_size, replace=False)
target_eval_indices = np.setdiff1d(np.arange(len(testset)), target_train_indices)

subset_tgt_train = torch.utils.data.Subset(trainset, target_train_indices)
subset_tgt_eval = torch.utils.data.Subset(trainset, target_eval_indices)

subset_tgt_train_loader = torch.utils.data.DataLoader(
    subset_tgt_train, batch_size=CFG.train_batch_size, shuffle=True, num_workers=2
)
subset_tgt_eval_loader = torch.utils.data.DataLoader(
    subset_tgt_eval, batch_size=CFG.val_batch_size, shuffle=False, num_workers=2
)

wandb.init(
    entity="cysec",
    project="membership_inference_attack",
    group=f"{shadow_model.__class__.__name__}_target",
    name=run_name,
)

train(
    CFG,
    target_model,
    subset_tgt_train_loader,
    subset_tgt_eval_loader,
    optimizer,
    CFG.save_path,
    shadow_number=-1,
    scheduler=None,
)

wandb.finish()
