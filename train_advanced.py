import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from advanced_model import ResNetEmotion

# Define paths for training and testing data
train_dir = 'C:/Users/Lenovo/Downloads/archive/train'
test_dir = 'C:/Users/Lenovo/Downloads/archive/test'

# Data transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels for ResNet
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

# Load the training dataset from the train directory
trainset = datasets.ImageFolder(root=train_dir, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Load the testing dataset from the test directory
testset = datasets.ImageFolder(root=test_dir, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Define the model, loss function, and optimizer
model = ResNetEmotion(num_classes=7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'model_resnet.pth')
