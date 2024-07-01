import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from .kar import Kar3
from .modules.hyperzzw import HyperZZW_L, HyperZZW_G
from argparse import ArgumentParser

# Command line argument parsing setup
parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)  # Set default batch size for training/evaluation
parser.add_argument("--mode", type=str, default="train")  # Set mode to either 'train' or 'eval'
args = parser.parse_args()

# Print the mode in which the script is running
if args.mode == "train":
    print("Started terminator in training mode")
elif args.mode == "eval":
    print("Started terminator in evaluation mode")
else:
    raise ValueError(f"Invalid mode: {args.mode} - Please use 'train' or 'eval'")

# Load the CIFAR dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Terminator model definition using PyTorch
class Terminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # TODO: Properly implement the HyperZZW architecture here
        
        self.resnet = resnet18(pretrained=True)  # ResNet-18 as the backbone
        self.resnet.fc = torch.nn.Identity()  # Remove the last fully connected layer
        self.hyperzzw_l = HyperZZW_L(torch.nn.Conv2d, in_channels=512, kernel_size=1)  # Local interaction module
        self.hyperzzw_g = HyperZZW_G  # Global interaction module
        self.linear = torch.nn.Linear(512, 10)  # Linear layer for classification

    def forward(self, x):
        x = self.resnet(x)  # Apply ResNet-18 backbone
        local_feat = self.hyperzzw_l(x, x)  # Apply local HyperZZW interaction
        global_feat = self.hyperzzw_g(x, x)  # Apply global HyperZZW interaction
        features = local_feat + global_feat  # Combine local and global features
        output = self.linear(features.mean(dim=[2, 3]))  # Apply linear layer and classify
        return output

# Initialize model, optimizer (Kar3 algorithm), and loss function
model = Terminator()
optimizer = Kar3(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Training or evaluation loop
if args.mode == "train":
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            images, labels = batch
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluation after training
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.4f}")
else:
    # Only evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")