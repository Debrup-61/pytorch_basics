# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Device

device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print(device)

# Get the MNIST dataset
train_data = datasets.MNIST("MNIST/", train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST("MNIST/", train=False, transform=transforms.ToTensor(), download=True)


# Hyperparameter
batch_size = 64
input_size = (28*28)
num_classes = 10
learning_rate = 0.001
no_epochs = 2


# Get the Train DataLoader
train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=True)


class NN(nn.Module):

  def __init__(self, input_size, num_classes):
      super(NN, self).__init__()
      self.fc1 = nn.Linear(in_features=input_size, out_features=50, device=device)
      self.fc2 = nn.Linear(in_features=50, out_features=num_classes, device=device)

  def forward(self,x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      return x


# Initialize Model
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Set the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training of the model

for epoch in range(no_epochs):
  for batch_index,(data,targets) in enumerate(train_loader):

      # Reshape the input data to 1 dim vector keeping batch_size same
      data = data.to(device)
      targets = targets.to(device)

      data = data.reshape(data.shape[0], -1)

      scores = model(data)
      loss = criterion(scores, targets)

      # Enter training
      model.train()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


def check_accuracy(loader,model):
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

        x = x.reshape(x.shape[0], -1)

        scores = model(x)

        _, predictions = torch.max(scores, 1)

        n_correct += (predictions == y).sum()
        n_samples += predictions.shape[0]

    print(f"Got {n_correct}/{n_samples} , Accuracy = {float(n_correct)/float(n_samples)*100:.2f}")
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)








































