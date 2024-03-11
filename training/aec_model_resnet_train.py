#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# PATH_DATA = '~/Documents/tu_dortmund/project_group/data/train'
PATH_DATA = '~/codebase-v1/data/data/train'
# PATH_SAVE = '~/Documents/tu_dortmund/project_group/'
PATH_SAVE = '~/codebase-v1/'

# Hyperparamers
num_epochs = 1
batch_size = 10
learning_rate = 0.001

# load the pre-trained Resnet-18 model
resnet18 = models.resnet18(pretrained=True)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class Autoencoder(nn.Module):
    def __init__(self, encoder):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, 
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Create an instance of the autoencoder
autoencoder = Autoencoder(resnet18)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Define your dataset and dataloader
train_dataset = datasets.ImageFolder(PATH_DATA, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Set the device for training (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using - ", device)
autoencoder = autoencoder.to(device)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (images, _) in enumerate(train_loader):
        # Move the images to the device
        images = images.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = autoencoder(images)

        # Compute the loss
        loss = criterion(outputs, images)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss for the epoch
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, running_loss / len(train_loader)))


# Save the trained model
torch.save(autoencoder.state_dict(), 'autoencoder_resnet.pth')