from utils.seed import seed_everything
import pandas as pd
import numpy as np
import yaml
from easydict import EasyDict
from joblib import dump, load

# get classifier models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])
    CFG_ATTACK = EasyDict(SAVED_CFG["CFG_ATTACK"])

# seed for future replication
seed_everything(CFG.seed)

# load model from the path
model_loaded = load(CFG_ATTACK.attack_model_path)
