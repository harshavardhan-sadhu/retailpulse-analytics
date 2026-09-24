import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

torch.manual_seed(42)

ts = pd.read_csv('data/processed/daily_sales_timeseries.csv')
ts.columns = ['ds', 'y']
ts['ds'] = pd.to_datetime(ts['ds'])

train_ts = ts.iloc[:-30].copy()
test_ts = ts.iloc[-30:].copy()

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_ts[['y']])
test_scaled = scaler.transform(test_ts[['y']])

look_back = 14

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

full_scaled = np.concatenate([train_scaled, test_scaled])
X_all, y_all = create_sequences(full_scaled, look_back)

split_idx = len(train_scaled) - look_back
X_train, y_train = X_all[:split_idx], y_all[:split_idx]
X_test, y_test = X_all[split_idx:], y_all[split_idx:]

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

print(f"Train sequences: {X_train_t.shape}, Test sequences: {X_test_t.shape}")

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMForecaster()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    predictions_scaled = model(X_test_t).numpy()

predictions = scaler.inverse_transform(predictions_scaled)
actuals = scaler.inverse_transform(y_test_t.numpy())

mape_mask = actuals.flatten() > 0
mape = np.mean(np.abs((actuals.flatten()[mape_mask] - predictions.flatten()[mape_mask]) / actuals.flatten()[mape_mask])) * 100

print(f"\nLSTM MAPE: {mape:.2f}%")
print(f"Prophet baseline MAPE: 24.06%")

torch.save(model.state_dict(), 'models/lstm_model.pth')

plt.figure(figsize=(14, 6))
plt.plot(test_ts['ds'].iloc[-len(actuals):], actuals, label='Actual', marker='o')
plt.plot(test_ts['ds'].iloc[-len(predictions):], predictions, label='LSTM Forecast', marker='x')
plt.title(f'LSTM Forecast vs Actual (MAPE: {mape:.2f}%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('docs/lstm_forecast_vs_actual.png', dpi=100, bbox_inches='tight')

print("Done - LSTM model trained and saved")
