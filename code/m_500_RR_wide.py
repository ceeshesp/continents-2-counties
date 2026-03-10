import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

start = datetime.now()

print('Started:', start)

# --- read data
df = pd.read_csv('c:/idx/mos/m_500_wide.csv')

X = df[[f'x_{i}' for i in range(500)]].values
y = df['leb'].values

# --- prepare OOS prediction array
oos_pred = np.zeros(len(df))

# --- 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y[train_idx]

    model = Ridge(alpha=0.001)
    model.fit(X_train, y_train)

    oos_pred[test_idx] = model.predict(X_test)

# --- compute metrics (global OOS)
mse = mean_squared_error(y, oos_pred)
r2 = r2_score(y, oos_pred)
mad = mean_absolute_error(y, oos_pred)

# --- build results table
results = df[['loc', 'code', 'lvl', 'leb']].copy()
results['leb'] = results['leb'] * 100
results['pred'] = oos_pred * 100
results['abse'] = np.abs(results['leb'] - results['pred'])
results['abse_pct'] = results['abse'] / results['leb'] * 100

# --- save
results.to_csv('c:/idx/mos/m_500_RR_wide.csv', index=False)

end = datetime.now()

print('Finished:', end)
print('Duration:', end - start)
print("OOS MSE:", mse)
print("OOS R2:", r2)
print("OOS MAD:", mad)


