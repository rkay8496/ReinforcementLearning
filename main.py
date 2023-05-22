import torch
import numpy as np
import matplotlib.pyplot as plt

# 임의의 함수로서 np.sin을 사용
x = np.linspace(-2*np.pi, 2*np.pi, 200)
y = np.sin(x)

# PyTorch Tensor로 변환
x_torch = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_torch = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 단일 은닉층을 가진 신경망 정의
model = torch.nn.Sequential(
    torch.nn.Linear(1, 50),
    torch.nn.Sigmoid(),
    torch.nn.Linear(50, 1),
)

# 손실 함수 및 옵티마이저 정의
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 모델 학습
for t in range(5000):
    y_pred = model(x_torch)
    
    loss = loss_fn(y_pred, y_torch)
    if t % 1000 == 999:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 결과 시각화
plt.figure(figsize=(8,5))
plt.plot(x, y, label='True function')
plt.plot(x, model(x_torch).detach().numpy(), label='Approximated function')
plt.legend()
plt.show()
