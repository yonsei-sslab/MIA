# -*- coding: utf-8 -*-
from shadow.make_data import make_member_nonmember
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from utils.seed import seed_everything
from utils.load_config import load_config
import pandas as pd
import numpy as np
import yaml
from easydict import EasyDict
from joblib import dump, load
import importlib

# get metric and train, test support
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# get classifier models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


# load config
CFG = load_config("CFG")
CFG_ATTACK = load_config("CFG_ATTACK")

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

testset = DSET_CLASS(root="./data", train=False, download=True, transform=transform)
trainset = DSET_CLASS(root="./data", train=True, download=True, transform=transform)

print("mapped classes to ids:", testset.class_to_idx)
columns_attack_sdet = [f"top_{index}_prob" for index in range(CFG.topk_num_accessible_probs)]

# load member data
list_nonmember_indices = pd.read_csv("./attack/train_indices.csv")["index"].to_list()
list_member_indices = np.random.choice(len(testset), len(list_nonmember_indices), replace=False)

subset_nonmember = Subset(trainset, list_nonmember_indices)
subset_member = Subset(testset, list_member_indices)

subset_nonmember_loader = DataLoader(
    subset_nonmember, batch_size=CFG.train_batch_size, shuffle=True, num_workers=2
)

subset_member_loader = DataLoader(
    subset_member, batch_size=CFG.train_batch_size, shuffle=True, num_workers=2
)

# target model loading (equivalent to API model that yields the prediction and logit)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_architecture = importlib.import_module("torchvision.models")
model_class = getattr(model_architecture, CFG.model_architecture)
target_model = model_class(pretrained=CFG.bool_pretrained)
target_model.fc = nn.Linear(in_features=target_model.fc.in_features, out_features=CFG.num_classes)
target_model.to(device)
target_model.load_state_dict(torch.load(CFG_ATTACK.target_model_path))
target_model.eval()

# target model request with member and nonmember data
member_dset, non_member_dset = make_member_nonmember(
    target_model, subset_member_loader, subset_nonmember_loader, nn.CrossEntropyLoss(), device
)
df_member = pd.DataFrame(member_dset, columns=columns_attack_sdet)
df_member["is_member"] = 1
df_non_member = pd.DataFrame(non_member_dset, columns=columns_attack_sdet)
df_non_member["is_member"] = 0
df_target_inference = pd.concat([df_member, df_non_member])

# load model from the path
if "cat" in CFG_ATTACK.attack_model_path.lower():
    attack_model = CatBoostClassifier()
    attack_model.load_model(CFG_ATTACK.attack_model_path)
else:
    attack_model = load(CFG_ATTACK.attack_model_path)
X_test = df_target_inference[columns_attack_sdet].to_numpy()
y_true = df_target_inference["is_member"].to_numpy()
y_pred = attack_model.predict(X_test)

# get accuracy, precision, recall, f1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
accuracy = accuracy_score(y_true, y_pred)
print("precision:", precision)
print("recall:", recall)
print("f1-score:", f1_score)
print("accuracy:", accuracy)
