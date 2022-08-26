from utils.seed import seed_everything
from utils.load_config import load_config
import pandas as pd
import numpy as np

# get metric and train, test support
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve

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

df_shadow = pd.read_csv(CFG_ATTACK.attack_dset_path)
print("Reading attack dataset:", CFG_ATTACK.attack_dset_path)
print(df_shadow.head)
print("data shape:", df_shadow.shape)

# train attack model
y = df_shadow["is_member"]
X = df_shadow.drop(["is_member"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=CFG_ATTACK.test_size, random_state=CFG.seed
)


# fit attack model: https://github.com/snoop2head/ml_classification_tutorial/blob/main/ML_Classification.ipynb
# model = xgb.XGBClassifier(n_estimators=CFG_ATTACK.n_estimators, n_jobs=-1, random_state=CFG.seed)
# model = lgb.LGBMClassifier(n_estimators=CFG_ATTACK.n_estimators, n_jobs=-1, random_state=CFG.seed)
model = CatBoostClassifier(
    iterations=200, depth=2, learning_rate=0.25, loss_function="Logloss", verbose=True
)  # https://catboost.ai/en/docs/concepts/loss-functions-classification

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test, model.predict(X_test), average="binary"
)
print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1_score:", f1_score)

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
print("fpr:", np.mean(fpr))
print("tpr:", np.mean(tpr))
print("thresholds:", thresholds)
#

save_path = f"./attack/{model.__class__.__name__}_{accuracy}"
# dump(model, save_path)
model.save_model(save_path)

