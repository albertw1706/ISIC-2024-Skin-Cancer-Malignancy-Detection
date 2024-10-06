# Define a function to train the model
import lightgbm as lgb
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
from optuna import Trial, visualization
import os
from sklearn.metrics import roc_auc_score

def score(solution: np.ndarray, submission: np.ndarray, min_tpr: float=0.80) -> float:
    v_gt = abs(solution-1)
    v_pred = np.array([1.0 - x for x in submission])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return(partial_auc)

def pauc_80(preds, data):
    score_value = score(data.get_label(), preds, min_tpr=0.8)
    return 'pauc_80', score_value, True
    
# Define the configuration space for Optuna
def Objective(trial):
    num_leaves = trial.suggest_categorical("num_leaves", [10, 15, 20, 25, 30])
    min_data_in_leaf = trial.suggest_categorical("min_data_in_leaf", [50, 60, 70, 80, 90])
    bagging_freq = trial.suggest_categorical("bagging_freq", [0, 1])  # Number of steps in the model
    feature_fraction = trial.suggest_categorical("feature_fraction", [0.7, 0.8, 0.9, 1])  # Relaxation parameter
    lambda_l1 = trial.suggest_categorical("lambda_l1", [0.25, 0.5, 0.75, 1])  # Shared GLU layers
    lambda_l2 = trial.suggest_categorical("lambda_l2", [0.1, 1, 10])  # Shared GLU layers
    num_boost_round = trial.suggest_categorical('num_boost_round', [400, 600, 800, 1000])

    # TabNet parameters
    lgb_params = {
    'objective': 'binary',
    'metric': 'none',  # Use a standard metric for evaluation
    'verbose': 0,
    'learning_rate': 0.01,  # Increase if model is converging too slowly
    'num_leaves': num_leaves,  # Reduce for simpler models
    'min_data_in_leaf': min_data_in_leaf,  # Increase to prevent overfitting
    'pos_bagging_fraction': 0.9,  # Adjust based on variance
    'neg_bagging_fraction': 0.05,  # Adjust based on variance
    'bagging_freq': bagging_freq,  # Reduce or disable if bagging is not helping
    'feature_fraction': feature_fraction,  # Increase to use more features
    'lambda_l1': lambda_l1,  # Keep low or 0 if L1 regularization is not needed
    'lambda_l2': lambda_l2  # Keep low or 0 if L2 regularization is not needed
} 

    # Train-Test Split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X.to_numpy(dtype=np.float32),
        y.to_numpy(dtype=np.float32),
        shuffle=True,
        test_size=0.2,  # 80% train, 20% validation
        random_state=42
    )

    train_df_dataset = lgb.Dataset(
    X_train,
    y_train,
    categorical_feature=df_test1_cols,
    free_raw_data=False
)
    
    val_df_dataset = lgb.Dataset(
    X_valid,
    y_valid,
    reference=train_df_dataset
)

    # Fitting the model with early stopping
    classifier = lgb.train(
        lgb_params,                  # Hyperparameters
        train_df_dataset,            # Training Dataset (correct argument)
        valid_sets=[val_df_dataset], # Validation Dataset for evaluation
        num_boost_round=num_boost_round,
        feval=pauc_80,               # Custom evaluation function
        callbacks=[
            lgb.early_stopping(stopping_rounds=40),
            lgb.log_evaluation(20)
        ]
    )

    # Append best validation score
    best_cost = classifier.best_cost
    if isinstance(best_cost, torch.Tensor):
        best_cost = best_cost.cpu().numpy()
    elif not isinstance(best_cost, np.ndarray):
        best_cost = np.array(best_cost)
        
    return best_cost

if __name__ == '__main__':

    # Setting up features and target
    df_raw = pd.read_csv("train-metadata.csv", low_memory=False)
    df_test1 = df_raw.drop(['patient_id', 'age_approx', 'sex', 'image_type', 'tbp_lv_Aext', 'tbp_lv_Bext', 'tbp_lv_Cext', 'tbp_lv_Hext', 'tbp_lv_Lext', 'tbp_lv_stdLExt', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_location', 'tbp_lv_location_simple', 'tbp_lv_radial_color_std_max', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', 'attribution', 'copyright_license', 'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 'mel_mitotic_index', 'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence', 'tbp_lv_deltaLB'], axis=1)
    df_test1['anatom_site_general'] = df_test1['anatom_site_general'].fillna('Unknown')
    df_test1['anatom_site_general'] = df_test1['anatom_site_general'].replace({'lower extremity': 0, 'head/neck': 1, 'posterior torso': 2, 'anterior torso': 3, 'upper extremity': 4, 'Unknown': 5}, regex=True).astype('int32')
    df_test1['tbp_tile_type'] = df_test1['tbp_tile_type'].replace({'3D: white': 0, '3D: XP': 1}, regex=True).astype('int32')
    df_test1_cols = ['anatom_site_general', 'tbp_tile_type']

    data = df_test1.columns.values.tolist()
    print (data)

    exclude = ['isic_id', 'target']
    input = [i for i in data if i not in exclude]

    X = df_test1[input]
    y = df_test1['target']

    study = optuna.create_study(direction="maximize", study_name='LGB optimization')

    timeout = 15 * 60 * 60 

    study.optimize(Objective, timeout=timeout) 


## Correlation analysis
corr = df_test1[input].corr().abs()
mask = np.triu(np.ones_like(corr, dtype=np.bool))

fig, ax = plt.subplots(figsize=(14, 14))

# plot heatmap
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            cbar_kws={"shrink": .8})
# yticks
plt.yticks(rotation=0)
plt.show()


import torch
import sys
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")