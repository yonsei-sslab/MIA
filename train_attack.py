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

df_shadow = pd.read_csv(CFG_ATTACK.attack_dset_path)

# train attack model
y = df_shadow["is_member"]
X = df_shadow.drop(["is_member"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=CFG_ATTACK.test_size, random_state=CFG.seed
)


# fit model: https://github.com/snoop2head/ml_classification_tutorial/blob/main/ML_Classification.ipynb
model = xgb.XGBClassifier(n_estimators=CFG_ATTACK.n_estimators, n_jobs=-1, random_state=CFG.seed)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(accuracy)
save_path = f"./attack/{model.__class__.__name__}_{accuracy}.joblib"
dump(model, save_path)

