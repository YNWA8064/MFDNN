import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from torchvision import datasets
from torchvision.transforms import transforms

import matplotlib.pyplot as plt

# Make sure to use only 10% of the available MNIST data.
# Otherwise, experiment will take quite long (around 90 minutes).

train_set = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)

idx = torch.randperm(len(train_set))[:6000]
train_set.data = train_set.data[idx]
train_set.targets = torch.randint(0, 10, size=(6000,))


# (Modified version of AlexNet)
class AlexNet(nn.Module):
    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc_layer1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6400, 800),
            nn.ReLU(inplace=True),
            nn.Linear(800, 10)
        )

    def forward(self, x):
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = torch.flatten(output, 1)
        output = self.fc_layer1(output)
        return output


learning_rate = 0.1
batch_size = 64
epochs = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size,shuffle=True)


tick = time.time()
Train_accuracy, Train_loss = [], []
for epoch in range(150):
    print(f"\nEpoch {epoch + 1} / {epochs}")
    tl = 0
    ta = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = loss_function(out, labels)
        loss.backward()

        optimizer.step()
        with torch.no_grad():
            ta += torch.sum(torch.max(out, dim=1).indices == labels).item()
            tl += loss.item()
    Train_accuracy.append(ta/60)
    Train_loss.append(tl)
    print(f"Train Loss: {tl:.4f}")
    print(f'Train accuracy: {ta/60:.2f}%')

tock = time.time()
print(f"Total training time: {tock - tick}")

fig, ax1 = plt.subplots()
ax1.plot(list(range(150)), Train_accuracy, color='green', label='Train Accuracy')
ax1.set_ylabel('Accuracy')
ax1.tick_params('y')

ax2 = ax1.twinx()
ax2.plot(list(range(150)), Train_loss, color='red', label='Train Loss')
ax2.set_ylabel('Loss')
ax2.tick_params('y')

plt.xlabel('Epochs')
fig.legend()
plt.show()