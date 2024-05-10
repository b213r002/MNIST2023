import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def setup_torch_seed(seed=1):
    # pytorchに関連する乱数シードの固定を行う．
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 乱数シードを固定
setup_torch_seed()

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 畳み込み層の定義
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 全結合層の定義
        self.fc1 = nn.Linear(7*7*32, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 10)
        # プーリング層の定義
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        y = self.pool(F.relu(self.conv1(x)))
        y = self.pool(F.relu(self.conv2(y)))
        y = y.view(y.size()[0], -1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y

def train(model, train_loader, criterion, optimizer, device):
    
    # ネットワークモデルを学習モードに設定
    model.train()

    sum_loss = 0.0
    count = 0

    for data, label in train_loader:
        count += len(label)
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    
    return sum_loss/count

def test(model, test_loader, criterion, device):

    # ネットワークモデルを評価モードに設定
    model.eval()

    sum_loss = 0.0
    count = 0
    correct = 0

    with torch.no_grad():
        for data, label in test_loader:
            count += len(label)
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            loss = criterion(outputs, label)
            sum_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += torch.sum(pred == label)
    
    accuracy_rate = (correct / count).cpu().detach()

    return sum_loss/count, accuracy_rate

#訓練データ
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download = True)
#検証データ
test_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download = True)

batch_size = 256

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
next(iter(train_loader))[0].shape

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device : ", device)

model = MNIST_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)
model.to(device)
criterion.to(device)
print("model : ", model)
print("criterion : ", criterion)
print("optimizer : ", optimizer)

num_epoch = 10
train_loss_list = []
test_loss_list = []
accuracy_rate_list = []

for epoch in range(1, num_epoch+1, 1):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, accuracy_rate = test(model, test_loader, criterion, device)

    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    accuracy_rate_list.append(accuracy_rate)

    print("epoch : {}, train_loss : {}, test_loss : {}, accuracy_rate : {}".format(epoch, train_loss, test_loss, accuracy_rate))

import matplotlib.pyplot as plt
plt.plot(range(1, len(train_loss_list)+1, 1), train_loss_list, c='b', label='train loss')
plt.plot(range(1, len(test_loss_list)+1, 1), test_loss_list, c='r', label='test loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()

import matplotlib.pyplot as plt
plt.plot(range(1, len(accuracy_rate_list)+1, 1), accuracy_rate_list, c='b', label='accuracy_rate')
plt.xlabel("epoch")
plt.ylabel("accuracy_rate")
plt.legend()
plt.grid()
plt.show()

model.to(device)
model.eval()
class_count_list = [0,0,0,0,0,0,0,0,0,0]
class_accuracy_rate_list = [0,0,0,0,0,0,0,0,0,0]

for i in range(len(test_dataset)):
    image, label = test_dataset[i]
    image = image.view(-1, 1, 28, 28).to(device)
    class_count_list[label] = class_count_list[label] + 1
    # 推論
    prediction_label = torch.argmax(model(image))
    if label == prediction_label:
        class_accuracy_rate_list[label] = class_accuracy_rate_list[label] + 1

for i in range(10):
    class_accuracy = class_accuracy_rate_list[i] / class_count_list[i]
    sum_accuracy = sum(class_accuracy_rate_list) / sum(class_count_list)
    print("class{} : {:.5f}  ( {} / {})".format(i, class_accuracy, class_accuracy_rate_list[i], class_count_list[i]))
print("sum_accuracy : {} ( {} / {})".format(sum_accuracy, sum(class_accuracy_rate_list), sum(class_count_list)))

model.eval()

import cv2
# テストデータセットの画像ファイル名とラベルの辞書を作成
test_image_labels = {
    '0.jpeg': 0,
    '1.jpeg': 1,
    '2.jpeg': 2,
    '3.jpeg': 3,
    '4.jpeg': 4,
    '5.jpeg': 5,
    '6.jpeg': 6,
    '7.jpeg': 7,
    '8.jpeg': 8,
    '9.jpeg': 9,
    # 他の画像ファイル名とラベルを追加する
}

plt.figure(figsize=(20, 10))
for i, (image_file, label) in enumerate(test_image_labels.items()):  # 画像ファイル名とラベルのペアを取得
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)  # 画像を読み込む
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 画像をテンソルに変換し、次元を追加
    image_tensor = image_tensor.to(device)  # デバイスに送る
    # 推論
    prediction_label = torch.argmax(model(image_tensor))
    ax = plt.subplot(5, 10, i+1)
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    ax.set_title('label : {}\n Prediction : {}'.format(label, prediction_label), fontsize=15)
plt.show()









