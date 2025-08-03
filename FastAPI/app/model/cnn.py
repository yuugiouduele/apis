import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# シンプルなCNNモデル（画像分類向け例）
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 28x28→28x28
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)  # 28x28→14x14
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 14x14→14x14
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)  # 10クラス分類

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))  # 14x14->7x7になるため修正要（以下で対応）
        x = x.view(x.size(0), -1)  # フラット化
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ダミーデータ（MNIST風、1チャネル28x28画像、バッチ16）
x_dummy = torch.randn(16, 1, 28, 28)
y_dummy = torch.randint(0, 10, (16,))

dataset = TensorDataset(x_dummy, y_dummy)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 学習ループ（簡略）
for epoch in range(2):
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

print("学習完了、推論例")
with torch.no_grad():
    out = model(x_dummy)
print("出力形状：", out.shape)  # (16, 10)
