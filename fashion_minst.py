import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import wandb

wandb.init(
    project="FashionMNIST",
    config={"epochs": 5, "batch_size": 128, "learning_rate": 0.001},
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the images (mean, std)
    ]
)

# Load the dataset
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)  # 28x28 pixels
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)  # 10 classes
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))

        return x


model = FashionMNISTModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    wandb.log({"loss": avg_loss})


model.eval()
all_preds = []
all_labels = []
val_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

avg_val_loss = val_loss / len(test_loader)
wandb.log({"validation_loss": avg_val_loss})
confusion_matrix = np.zeros((10, 10), dtype=int)

for true, pred in zip(all_labels, all_preds):
    confusion_matrix[true, pred] += 1

accuracies = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

for i, acc in enumerate(accuracies):
    print(f"Dokładność klasy {i}: {acc:.2f}")
    wandb.log({f"class_{i}_accuracy": acc})

artifact = wandb.Artifact("fashion_mnist_model", type="code")
artifact.add_file("/home/jk/Documents/studia/sieci/lab3/mldevops_exercise/fashion_minst.py")
wandb.log_artifact(artifact)

wandb.finish()
