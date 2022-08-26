from shadow.trainer import train
from shadow.make_data import make_member_nonmember
from utils.seed import seed_everything
from utils.load_config import load_config
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

# load config
CFG = load_config("CFG")

# conduct training
if not os.path.exists(CFG.save_path):
    os.makedirs(CFG.save_path)

# seed for future replication
seed_everything(CFG.seed)

# Load the CIFAR dataset
# CIFAR train is used for SHADOW MODEL train & evaluation whereas CIFAR test is used for TARGET MODEL train & evaluation
if CFG.dataset_name.lower() == "cifar10":
    DSET_CLASS = torchvision.datasets.CIFAR10
    CFG.num_classes = 10
elif CFG.dataset_name.lower() == "cifar100":
    DSET_CLASS = torchvision.datasets.CIFAR100
    CFG.num_classes = 100

transform = transforms.Compose(
    [
        transforms.Resize((CFG.input_resolution, CFG.input_resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
shadow_set = DSET_CLASS(root="./data", train=True, download=True, transform=transform)
shadow_loader = torch.utils.data.DataLoader(
    shadow_set, batch_size=CFG.train_batch_size, shuffle=True, num_workers=2
)

# define dataset for attack model that shadow models will generate
print("mapped classes to ids:", shadow_set.class_to_idx)
columns_attack_sdet = [f"top_{index}_prob" for index in range(CFG.topk_num_accessible_probs)]
df_attack_dset = pd.DataFrame({}, columns=columns_attack_sdet + ["is_member"])

# random subset for shadow model train & validation from the CIFAR shadow_set
list_train_loader = []
list_eval_loader = []
list_test_loader = []

# make random subset for shadow model train & validation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Define Devices
for _ in range(CFG.num_shadow_models):
    train_indices = np.random.choice(len(shadow_set), CFG.shadow_train_size, replace=False)
    eval_indices = np.setdiff1d(np.arange(len(shadow_set)), train_indices)
    eval_indices = np.random.choice(eval_indices, CFG.shadow_train_size, replace=False)
    test_indices = np.setdiff1d(
        np.arange(len(shadow_set)), np.concatenate((train_indices, eval_indices))
    )
    test_indices = np.random.choice(test_indices, CFG.shadow_train_size, replace=False)

    subset_train = torch.utils.data.Subset(shadow_set, train_indices)
    subset_eval = torch.utils.data.Subset(shadow_set, eval_indices)
    subset_test = torch.utils.data.Subset(shadow_set, test_indices)

    subset_train_loader = torch.utils.data.DataLoader(
        subset_train, batch_size=CFG.train_batch_size, shuffle=True, num_workers=2
    )

    subset_eval_loader = torch.utils.data.DataLoader(
        subset_eval, batch_size=CFG.val_batch_size, shuffle=False, num_workers=2
    )

    subset_test_loader = torch.utils.data.DataLoader(
        subset_test, batch_size=CFG.val_batch_size, shuffle=False, num_workers=2
    )

    list_train_loader.append(subset_train_loader)
    list_eval_loader.append(subset_eval_loader)
    list_test_loader.append(subset_test_loader)


# Training multiple shadow models
model_architecture = importlib.import_module("torchvision.models")
model_class = getattr(model_architecture, CFG.model_architecture)
criterion = nn.CrossEntropyLoss()

# iterate through predefined shadow_loaders and validationloaders
for shadow_number, shadow_loader in enumerate(tqdm(list_train_loader)):
    evalloader = list_eval_loader[shadow_number]
    testloader = list_test_loader[shadow_number]

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
        shadow_loader,
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
        finetuned_model, shadow_loader, testloader, criterion, device
    )

    df_member = pd.DataFrame(member_dset, columns=columns_attack_sdet)
    df_member["is_member"] = 1
    df_non_member = pd.DataFrame(non_member_dset, columns=columns_attack_sdet)
    df_non_member["is_member"] = 0

    df_attack_dset = pd.concat([df_attack_dset, df_member, df_non_member])
    df_attack_dset.to_csv(
        f"./attack/{shadow_model.__class__.__name__}_pretrained_{CFG.bool_pretrained}_num_shadow_{CFG.num_shadow_models}_CIFAR{CFG.num_classes}.csv",
        index=False,
    )

    # Prevent OOM error by deleting finetuned model and datasets
    shadow_model.cpu()
    del shadow_model, optimizer, shadow_loader, evalloader, testloader
    torch.cuda.empty_cache()
    wandb.finish()

