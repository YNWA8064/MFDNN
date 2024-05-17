import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt

lr = 0.001
batch_size = 100
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Step 1:
'''

# MNIST dataset
dataset = datasets.MNIST(root='./../DATA/mnist_data/',
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=False)

train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [50000, 10000])

test_dataset = datasets.MNIST(root='./../DATA/mnist_data/',
                              train=False, 
                              transform=transforms.ToTensor())

# KMNIST dataset, only need test dataset
anomaly_dataset = datasets.KMNIST(root='./../DATA/kmnist_data/',
                              train=False, 
                              transform=transforms.ToTensor(),
                              download=False)

# print(len(train_dataset))  # 50000
# print(len(validation_dataset))  # 10000
# print(len(test_dataset))  # 10000
# print(len(anomaly_dataset))  # 10000

'''
Step 2: AutoEncoder
'''
# Define Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = F.relu(self.fc3(x))
        return z

# Define Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 784)
    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        x = F.sigmoid(self.fc3(z))  # to make output's pixels are 0~1
        x = x.view(x.size(0), 1, 28, 28) 
        return x
    
'''
Step 3: Instantiate model & define loss and optimizer
'''
enc = Encoder().to(device)
dec = Decoder().to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)


'''
Step 4: Training
'''
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


train_loss_list = []

import time
start = time.time()
for epoch in range(epochs) :
    print("{}th epoch starting.".format(epoch+1))
    enc.train()
    dec.train()
    for batch, (images, _) in enumerate(train_loader) :
        images = images.to(device)
        z = enc(images)
        reconstructed_images = dec(z)
        
        optimizer.zero_grad()
        train_loss = loss_function(images, reconstructed_images)
        train_loss.backward()
        train_loss_list.append(train_loss.item())

        optimizer.step()

        print(f"[Epoch {epoch:3d}] Processing batch #{batch:3d} reconstruction loss: {train_loss.item():.6f}", end='\r')
end = time.time()
print("Time ellapsed in training is: {}".format(end - start))

# plotting train loss
plt.plot(range(1,len(train_loss_list)+1), train_loss_list, 'r', label='Training loss')
plt.title('Training loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
# plt.savefig('loss.png')
plt.show()

enc.eval()
dec.eval()

'''
Step 5: Calculate standard deviation by using validation set
'''
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size)

scores = []
for images, _ in validation_loader:
    images = images.to(device)
    gaps = (images - dec(enc(images)))**2
    scores += torch.sum(gaps, dim=[1, 2, 3]).tolist()
mean, std = np.average(scores), np.std(scores)


threshold = mean + 3 * std
print("threshold: ", threshold)


'''
Step 6: Anomaly detection (mnist)
'''
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

TYPE_1_ERROR, TYPE_2_ERROR = 0, 0
cnt = 0
fig = plt.figure(1, figsize=(15, 6))
fig.suptitle('Misclassified Figures', fontsize=16)
gg = []
for idx, (images, _) in enumerate(test_loader):
    images = images.to(device)
    recon_image = dec(enc(images))
    gaps = (images - recon_image) ** 2
    gaps = torch.sum(gaps, dim=[1, 2, 3])
    TYPE_1_ERROR += torch.sum(gaps > threshold).item()
    if cnt<=4:
        for i in range(len(gaps)):
            if gaps[i] > threshold:
                gg.append(gaps[i])
                ax = fig.add_subplot(2, 5, cnt + 1)
                plt.imshow(images[i].cpu().squeeze().detach())

                ax = fig.add_subplot(2, 5, cnt + 6)
                plt.imshow(recon_image[i].cpu().squeeze().detach())

                cnt += 1
            if cnt == 5:
                break

plt.show()

cnt = 0
fig = plt.figure(1, figsize=(15, 6))
fig.suptitle('Correctly Classified Figures', fontsize=16)
gg = []

for idx, (images, _) in enumerate(test_loader):
    if cnt==5:
        break
    images = images.to(device)
    recon_image = dec(enc(images))
    gaps = (images - recon_image) ** 2
    gaps = torch.sum(gaps, dim=[1, 2, 3])
    for i in range(len(gaps)):
        if gaps[i] <= threshold:
            gg.append(gaps[i])
            ax = fig.add_subplot(2, 5, cnt + 1)
            plt.imshow(images[i].cpu().squeeze().detach())

            ax = fig.add_subplot(2, 5, cnt+6)
            plt.imshow(recon_image[i].cpu().squeeze().detach())

            cnt+=1
        if cnt==5:
            break

plt.show()



'''
Step 7: Anomaly detection (kmnist)
'''
anomaly_loader = torch.utils.data.DataLoader(dataset=anomaly_dataset, batch_size=batch_size)
cnt = 0
fig = plt.figure(1, figsize=(15, 6))
fig.suptitle('Misclassified Figures', fontsize=16)
gg = []
for idx, (images, _) in enumerate(anomaly_loader):
    images = images.to(device)
    recon_image = dec(enc(images))
    gaps = (images - recon_image) ** 2
    gaps = torch.sum(gaps, dim=[1, 2, 3])
    TYPE_2_ERROR += torch.sum(gaps < threshold).item()
    if cnt<=4:
        for i in range(len(gaps)):
            if gaps[i] <= threshold:
                gg.append(gaps[i])
                ax = fig.add_subplot(2, 5, cnt + 1)
                plt.imshow(images[i].cpu().squeeze().detach())

                ax = fig.add_subplot(2, 5, cnt + 6)
                plt.imshow(recon_image[i].cpu().squeeze().detach())

                cnt += 1
            if cnt == 5:
                break

plt.show()

print(f'Type I error : {TYPE_1_ERROR}/{len(test_dataset)} ({TYPE_1_ERROR/len(test_dataset)*100.:.3f}%)')
print(f'Type II error : {TYPE_2_ERROR}/{len(anomaly_dataset)} ({TYPE_2_ERROR/len(anomaly_dataset)*100.:.3f})%')