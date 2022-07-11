from shadow.trainer import train
from shadow.make_data import make_member_nonmember
from utils.seed import seed_everything
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from easydict import EasyDict
import yaml
import wandb
import importlib

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])

# conduct training
if not os.path.exists(CFG.save_path):
    os.makedirs(CFG.save_path)

# seed for future replication
seed_everything(CFG.seed)

# Load the CIFAR dataset
# CIFAR train is used for shadow model train & evaluation
# CIFAR test is used for target model train & evaluation
if CFG.num_classes == 10:
    DSET_CLASS = torchvision.datasets.CIFAR10
elif CFG.num_classes == 100:
    DSET_CLASS = torchvision.datasets.CIFAR100

transform = transforms.Compose(
    [
        transforms.Resize((CFG.input_resolution, CFG.input_resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
trainset = DSET_CLASS(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=2
)

# define dataset for attack model that shadow models will generate
print("mapped classes to ids:", trainset.class_to_idx)
columns_attack_sdet = [f"top_{index}_prob" for index in range(CFG.topk_num_accessible_probs)]
df_attack_dset = pd.DataFrame({}, columns=columns_attack_sdet + ["is_member"])

# random subset for shadow model train & validation from the CIFAR trainset
list_train_loader = []
list_eval_loader = []

# make random subset for shadow model train & validation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Define Devices
for _ in range(CFG.num_shadow_models):
    train_indices = np.random.choice(len(trainset), CFG.shadow_train_size, replace=False)
    eval_indices = np.setdiff1d(np.arange(len(trainset)), train_indices)
    eval_indices = np.random.choice(eval_indices, CFG.shadow_train_size, replace=False)

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

# Training multiple shadow models
model_architecture = importlib.import_module("torchvision.models")
model_class = getattr(model_architecture, CFG.model_architecture)
criterion = nn.CrossEntropyLoss()

# iterate through predefined trainloaders and validationloaders
for shadow_number, trainloader in enumerate(tqdm(list_train_loader)):
    evalloader = list_eval_loader[shadow_number]

    # define shadow model to finetune on the CIFAR train dataset
    shadow_model = model_class(pretrained=CFG.bool_pretrained)
    shadow_model.fc = nn.Linear(
        in_features=shadow_model.fc.in_features, out_features=CFG.num_classes
    )
    shadow_model = shadow_model.to(device)

    run_name = f"{shadow_model.__class__.__name__}_shadow_{shadow_number}"

    wandb.init(
        entity="cysec",
        project="membership_inference_attack",
        group=f"{shadow_model.__class__.__name__}_shadow",
        name=run_name,
    )

    optimizer = AdamW(
        shadow_model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )

    # finetune shadow model (validation metrics are recorded on wandb)
    finetuned_model = train(
        CFG,
        shadow_model,
        trainloader,
        evalloader,
        optimizer,
        CFG.save_path,
        shadow_number,
        scheduler=None,
        criterion=criterion,
        device=device,
    )

    # create member dataset vs non-member dataset based on finetuned model
    member_dset, non_member_dset = make_member_nonmember(
        finetuned_model, trainloader, evalloader, criterion, device
    )

    df_member = pd.DataFrame(member_dset, columns=columns_attack_sdet)
    df_member["is_member"] = 1
    df_non_member = pd.DataFrame(non_member_dset, columns=columns_attack_sdet)
    df_non_member["is_member"] = 0

    df_attack_dset = pd.concat([df_attack_dset, df_member, df_non_member])
    df_attack_dset.to_csv(
        f"./attack/{shadow_model.__class__.__name__}_pretrained_{CFG.bool_pretrained}_num_shadow_{CFG.num_shadow_models}.csv",
        index=False,
    )

    # Prevent OOM error by deleting finetuned model and datasets
    shadow_model.cpu()
    del shadow_model, optimizer, trainloader, evalloader
    torch.cuda.empty_cache()
    wandb.finish()

