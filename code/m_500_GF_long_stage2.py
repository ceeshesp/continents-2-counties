# stage2_final_train.py

import pandas as pd

from datetime import datetime
from pytorch_tabular.models import GANDALFConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular import TabularModel
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

df = pd.read_csv('c:/idx/mos/m_500_long.csv')

print(f'Loaded {len(df):,} rows')
print(f'Unique locations: {df['location'].nunique()}')
print(f'Items per location (expected ~501): {df.groupby('location').size().iloc[0]}')

# ---------------------------------------------------------
# 2. Shared configuration (same as Stage 1)
# ---------------------------------------------------------

data_config = DataConfig(
    target=['nv'],
    continuous_cols=['nv'],
    categorical_cols=['location', 'item'],
    normalize_continuous_features=False
)

trainer_config = TrainerConfig(
    batch_size=2048,
    max_epochs=40,
    checkpoints=None,
    early_stopping=None
)

optimizer_config = OptimizerConfig()

model_config = GANDALFConfig(
    task='regression',
    gflu_stages=4,
    gflu_feature_init_sparsity=0.3,
    gflu_dropout=0.0,
    learning_rate=1e-3,
    target_range=[(0.0, 1.0)]
)

# ---------------------------------------------------------
# 3. Train final model on ALL data
# ---------------------------------------------------------

start = datetime.now()

print('\n=== Stage 2: Final Training Started ===')
print('Start time:', start)

tb = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

tb.fit(train=df)

# ---------------------------------------------------------
# 4. Predict on ALL rows
# ---------------------------------------------------------

pred_all = tb.predict(df)

# ---------------------------------------------------------
# 5. Extract only leb rows
# ---------------------------------------------------------

leb_mask = df['item'] == 'leb'

leb_true = df.loc[leb_mask, 'nv'].values
leb_pred = pred_all.loc[leb_mask, 'nv_prediction'].values

# ---------------------------------------------------------
# 6. Compute Stage‑2 fit metrics (leb only)
# ---------------------------------------------------------

mad_leb = (abs(leb_true - leb_pred)).mean()
mapd_leb = (abs(leb_true - leb_pred) / leb_true).mean() * 100
r2_leb = r2_score(leb_true, leb_pred)
mse = mean_squared_error(leb_true, leb_pred)

print('\n=== Stage-2 Fit Metrics (leb rows only) ===')
print(f'MAD  : {mad_leb:.6f} years')
print(f'MAPD : {mapd_leb:.3f} %')
print(f'R2 : {r2_leb:.6f}')
print(f'MSE : {mse:.6f}')

# ---------------------------------------------------------
# 7. Save final predictions
# ---------------------------------------------------------

leb_final = pd.DataFrame({
    'location': df.loc[leb_mask, 'location'].values,
    'leb': leb_true,
    'pred': leb_pred,
})

leb_final['code'] = df.loc[leb_mask, 'code'].values
leb_final['lvl'] = df.loc[leb_mask, 'lvl'].values
leb_final['abse'] = abs(leb_final['leb'] - leb_final['pred'])
leb_final['abse_pct'] = leb_final['abse'] / leb_final['leb'] * 100

leb_final.to_csv('c:/idx/mos/m_500_GF_long_st2.csv', index=False)

end = datetime.now()

print('End time:', end)
print('Duration:', end - start)
print('\n=== Stage 2 Complete ===')


