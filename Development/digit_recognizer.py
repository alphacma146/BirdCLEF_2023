# %%[markdown]
# picture=28pix $\times$ 28pix
# |label|pixel0|pixel1|...|pixel783|
# |:---:|:---:|:---:|:---:|:---:|
# |1|0|0|...|0|
#
# - **label** the digit that was drawn by the user
# - **pixelX** x is an integer between 0 and 783
# %%
# standard lib
import time
from pathlib import Path
# third party
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

train_data = pd.read_csv(
    Path("Data/digit-recognizer/train.csv")
)
test_data = pd.read_csv(
    Path("Data/digit-recognizer/test.csv")
)

# %%


def show_image(data: np.ndarray) -> None:
    """
    28*28にreshape可能なら画像を表示する
    """
    try:
        data.reshape(28, 28)
    except BaseException:
        print(f"Image broken!! {data.shape}")
        return

    digit_array = data.reshape(28, 28)
    sns.heatmap(digit_array, cbar=False)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()


cnt = 0
while cnt <= 5:
    n = np.random.randint(0, len(train_data) + 1)
    data = train_data[n:n + 1]
    print(f"{data.iloc[:,0:1]}")
    show_image(data.iloc[:, 1:].values)
    cnt += 1
# %%


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # 特徴量検出
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(6),  # バッチ正規化
            nn.ReLU(inplace=True),

            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 分類
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 120),
            nn.BatchNorm1d(120),  # 1次元に注意
            nn.ReLU(inplace=True),

            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(inplace=True),

            nn.Linear(84, 10),
            nn.BatchNorm1d(10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)


# %%
# データ前処理
x_train = train_data.iloc[:, 1:].values.astype('float32')
y_train = train_data.iloc[:, 0].values.astype('int32')
x_test = test_data.values.astype('float32')
x_train = x_train.reshape(x_train.shape[0], 28, 28)
# train と validation に分割
(
    train_image,
    validation_image,
    train_label,
    validation_label
) = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)
# ndarray形式からTensor形式へ変換
train_image = torch.from_numpy(train_image)
validation_image = torch.from_numpy(validation_image)
train_label = torch.from_numpy(train_label).type(torch.LongTensor)
validation_label = torch.from_numpy(validation_label).type(torch.LongTensor)
# バッチサイズとエポック数の定義
batch_size = 100
num_epochs = 30
# データローダにデータを読み込ませる
train = torch.utils.data.TensorDataset(train_image, train_label)
validation = torch.utils.data.TensorDataset(validation_image, validation_label)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(
    validation,
    batch_size=batch_size,
    shuffle=False
)
# %%
# モデルインスタンス作成
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# 損失関数の定義(cross entropy)
criterion = nn.CrossEntropyLoss()

# オプティマイザの定義(今回はSGD)
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# %%
# 学習

# 学習過程でロスと精度を保持するリスト
train_losses, val_losses = [], []
train_accu, val_accu = [], []

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    # 学習用データで学習
    train_loss = 0
    correct = 0
    model.train()
    for images, labels in train_loader:
        # to でGPUを利用するように指定
        images = images.to(device)
        labels = labels.to(device)

        # 勾配の初期化
        optimizer.zero_grad()

        # 順伝播
        outputs = model(torch.unsqueeze(images, 1))
        # unsqueezeは次元エラーを回避するために使用する。次元を1つ増やせる。

        # ロスの計算と逆伝播
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 正答数を計算
        predicted = torch.max(outputs.data, 1)[1]
        correct += (predicted == labels).sum()

        train_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    train_accu.append(correct.item() / len(train_loader))

    # 検証用データでlossと精度の計算
    val_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in validation_loader:
            # to でGPUを利用するように指定
            images = images.to(device)
            labels = labels.to(device)

            # 順伝播
            outputs = model(torch.unsqueeze(images, 1))

            # ロスの計算
            val_loss += criterion(outputs, labels).item()

            # 正答数を計算
            predicted = torch.max(outputs.data, 1)[1]
            correct += (predicted == labels).sum()

    val_losses.append(val_loss / len(validation_loader))
    val_accu.append(correct.item() / len(validation_loader))

    print(
        f"Epoch: {epoch+1}/{num_epochs}",
        f"Time: {time.time()-epoch_start_time:.2f}s",
        f"Training Loss: {train_losses[-1]:.3f}",
        f"Training Accu: {train_accu[-1]:.3f}",
        f"Val Loss: {val_losses[-1]:.3f}",
        f"Val Accu: {val_accu[-1]:.3f}",
        sep=" "
    )
# %%
# lossの可視化
sns.set_style("whitegrid", {"grid.color": ".2", "grid.linestyle": "--"})
plt.figure(figsize=(10, 6))
plt.ylabel("Loss")
sns.lineplot(
    y=train_losses, x=np.arange(1, len(train_losses) + 1),
    label='Training Loss',
    linewidth=5, color="red", alpha=1, zorder=1
)
sns.lineplot(
    y=val_losses, x=np.arange(1, len(val_losses) + 1),
    label='Validation Loss',
    linewidth=5, color="blue", zorder=2
)
plt.show()
# accuracyの可視化
plt.figure(figsize=(10, 6))
plt.ylabel("Accuracy")
sns.lineplot(
    y=train_accu, x=np.arange(1, len(train_accu) + 1),
    label='Training Accuracy',
    linewidth=5, color="red", alpha=1, zorder=1
)
sns.lineplot(
    y=val_accu, x=np.arange(1, len(val_accu) + 1),
    label='Validation Accuracy',
    linewidth=5, color="blue", zorder=2
)
plt.show()

torch.save(model.state_dict(), Path("digit-cnn.pth"))
# %%
# テストデータを使って回答を作る
x_test = x_test.reshape(x_test.shape[0], 28, 28)
x_test = torch.from_numpy(x_test).to(device)
model.eval()
output = model(torch.unsqueeze(x_test, 1))
output = output.to('cpu')
prediction = pd.DataFrame(
    data={
        "ImageId": np.arange(1, len(x_test) + 1),
        "Label": torch.argmax(output, 1)
    }
)
prediction.to_csv(
    Path("digit-recognizer/result_submission.csv"),
    index=False
)
# %%
cnt = 0
valid_train = train_data.iloc[:, 1:].values.astype('float32')
valid_train = valid_train.reshape(valid_train.shape[0], 28, 28)
while cnt <= 10:
    n = np.random.randint(0, len(valid_train) + 1)
    data = valid_train[n:n + 1]
    img = torch.from_numpy(data).to(device)
    predict = model(torch.unsqueeze(img, 1))
    res = torch.argmax(predict, 1).item(), train_data['label'][n]
    tens = pd.DataFrame(
        np.round(predict.tolist(), decimals=3),
        index=["Tensor"]
    )
    if not (res_bool := res[0] == res[1]):
        print(
            f"index: {n}",
            f"Predict: {res[0]}\tLabel: {res[1]}\t{res_bool}",
            f"{tens}",
            sep="\n"
        )
        show_image(data)
        cnt += 1

# %%
