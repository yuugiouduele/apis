import torch
import matplotlib.pyplot as plt

# 例: 単純な線形モデル
model = torch.nn.Linear(10, 1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

params = count_parameters(model)
print(f"モデルの学習可能パラメータ数: {params}")

# 仮の損失値リスト(エポック毎)
losses = [1.0/(epoch+1) for epoch in range(10)]
plt.plot(range(1, len(losses)+1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("学習曲線")
plt.show()

# 収束性の簡易判定（損失減少傾向）
def check_convergence(losses):
    return all(earlier >= later for earlier, later in zip(losses, losses[1:]))

print("収束しているか？", check_convergence(losses))
