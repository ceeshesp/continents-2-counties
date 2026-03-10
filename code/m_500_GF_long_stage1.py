# stage1_oos_eval_groupkfold.py

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from pytorch_tabular.models import GANDALFConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular import TabularModel
from datetime import datetime

start = datetime.now()

print('Started:', start)

# --- load long-format data
df = pd.read_csv('c:/idx/mos/m_500_long.csv')

groups = df['location'].values
unique_locs = df['location'].unique()
n_locs = len(unique_locs)
print(f'Number of unique locations: {n_locs}')

# --- model configs (same as Stage 2)
data_config = DataConfig(
    target=['nv'],
    continuous_cols=['nv'],
    categorical_cols=['location', 'item'],
    normalize_continuous_features=False,
)

trainer_config = TrainerConfig(
    batch_size=2048,
    max_epochs=40,
    checkpoints=None,
    early_stopping=None,
)

optimizer_config = OptimizerConfig()

model_config = GANDALFConfig(
    task='regression',
    gflu_stages=4,
    gflu_feature_init_sparsity=0.3,
    gflu_dropout=0.0,
    learning_rate=1e-3,
    target_range=[(0.0, 1.0)],
)

# --- storage for OOS predictions (only for leb rows)
oos_pred = np.full(len(df), np.nan)

# K-fold grouped CV on locations
K = 5
gkf = GroupKFold(n_splits=K)

fold = 0
for train_idx, test_idx in gkf.split(df, groups=groups):
    fold += 1
    print(f'\n=== Fold {fold}/{K} ===')
    fold_locs = df.iloc[test_idx]['location'].unique()
    print(f'Test locations in this fold: {len(fold_locs)}')

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[test_idx].copy()

    tb = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    tb.fit(train=train_df, validation=val_df)

    pred_df = tb.predict(val_df)

    # store predictions only for leb rows in this fold
    leb_mask = val_df['item'] == 'leb'
    test_leb_idx = test_idx[leb_mask]

    oos_pred[test_leb_idx] = pred_df.loc[leb_mask, 'nv_prediction'].values

# --- now compute OOS metrics over ALL leb rows
leb_mask_all = df['item'] == 'leb'
leb_true = df.loc[leb_mask_all, 'nv'].values
leb_oos_pred = oos_pred[leb_mask_all]

# sanity check: no NaNs
n_nan = np.isnan(leb_oos_pred).sum()

print(f'\nNaNs in OOS leb predictions: {n_nan}')
if n_nan > 0:
    raise RuntimeError('Some leb rows did not receive OOS predictions.')

mse = mean_squared_error(leb_true, leb_oos_pred)
r2 = r2_score(leb_true, leb_oos_pred)
mad = np.mean(np.abs(leb_true - leb_oos_pred))
mapd = np.mean(np.abs(leb_true - leb_oos_pred) / leb_true) * 100

print('\n=== Leakage-free OOS metrics (GroupKFold) ===')
print(f'MSE : {mse:.6f}')
print(f'R2  : {r2:.6f}')
print(f'MAD : {mad:.6f} (years)')
print(f'MAPD: {mapd:.3f} %')

# --- save OOS evaluation results for leb only
eval_df = df[leb_mask_all].copy()
eval_df['pred'] = leb_oos_pred
eval_df['abse'] = np.abs(eval_df['nv'] - eval_df['pred'])
eval_df['abse_pct'] = eval_df['abse'] / eval_df['nv'] * 100

eval_df.to_csv('c:/idx/mos/m_500_GF_long_st1.csv', index=False)

end = datetime.now()

print('Finished:', end)
print('Duration:', end - start)

print('OOS MSE:', mse)
print('OOS R2:', r2)
