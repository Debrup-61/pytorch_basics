import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(kernel_size=(3, 3), in_channels=n_channels, stride=1, padding=(1, 1), out_channels=8)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=8, stride=1, padding=(1, 1), out_channels=16)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(16*7*7, n_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# Device

device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
# device = torch.device("cpu")
print(device)

# Get the MNIST dataset
train_data = datasets.MNIST("MNIST/", train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST("MNIST/", train=False, transform=transforms.ToTensor(), download=True)

print("No of training images", len(train_data))
print("No of test set images", len(test_data))

# Hyperparameter
batch_size = 64
num_channels = 1
num_classes = 10
learning_rate = 0.001
no_epochs = 3


# Get the Train DataLoader
train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=True)

# Example of an image in dataset
plt.figure(figsize=(4, 4))
sample_idx = 16
img, _ = train_data[sample_idx]
plt.imshow(img.permute(1, 2, 0), cmap="gray")
plt.show()

# Initialize CNN Model
#print("Defined CNN")
model = CNN(n_channels=num_channels, n_classes=num_classes)
model = model.to(device)

#print("Got model")

# Set the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#print("Got optimizer")

# Training of the model
train_loss = []
epoch_accuracy = []
print("Start training")
for epoch in range(no_epochs):
    n_correct = 0
    n_sample = 0
    train_loss.append([])

    for batch_index, (data, targets) in enumerate(train_loader):
        #print("Inside mini-batch")
        data = data.to(device)
        targets = targets.to(device)

        #print("Forward")
        scores = model(data)
        #print("Got scores")
        _, preds = torch.max(scores, 1)

        #print(preds[0])
        n_correct += (preds == targets).sum()
        n_sample += preds.shape[0]

        loss = criterion(scores, targets)
        train_loss[epoch].append(loss)
        # Enter training
        model.train()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = float(n_correct) / float(n_sample) * 100
    print(f"Training Accuracy in epoch{epoch+1} = {acc:.2f}")
    epoch_accuracy.append(acc)

def check_accuracy(loader, model):
    model.eval()

    if loader.dataset.train == True:
        print("Evaluation on train dataset")
    else:
        print("Evaluation on test dataset")

    n_correct = 0
    n_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        scores = model(x)

        _, predictions = torch.max(scores, 1)

        n_correct += (predictions == y).sum()
        n_samples += predictions.shape[0]

    print(f"Got {n_correct}/{n_samples} , Accuracy = {float(n_correct)/float(n_samples)*100:.2f}")
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

















