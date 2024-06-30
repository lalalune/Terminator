import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from .kar import Kar3
from .modules.hyperzzw import HyperZZW_L, HyperZZW_G
from argparse import ArgumentParser

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
# mode train or eval
parser.add_argument("--mode", type=str, default="train")
args = parser.parse_args()

if args.mode == "train":
    print("Started terminator in training mode")
elif args.mode == "eval":
    print("Started terminator in evaluation mode")
else:
    raise ValueError(f"Invalid mode: {args.mode} - Please use 'train' or 'eval'")


# Load a toy dataset from Hugging Face
dataset = load_dataset("rotten_tomatoes", split="train")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create a DataLoader
dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

# Define your Terminator model
class Terminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(tokenizer.vocab_size, 128)
        self.hyperzzw_l = HyperZZW_L(torch.nn.Conv1d, in_channels=128, kernel_size=1)
        self.hyperzzw_g = HyperZZW_G
        self.linear = torch.nn.Linear(128, 2)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        print(f"Embedded input shape: {x.shape}")
        assert x.ndim == 3, f"Expected embedded input to have 3 dimensions, got {x.ndim}"
        assert x.size(2) == 128, f"Expected embedding dimension to be 128, got {x.size(2)}"

        local_feat = self.hyperzzw_l(x, x)
        global_feat = self.hyperzzw_g(x, x)

        features = local_feat + global_feat
        print(f"Combined features shape: {features.shape}")
        assert features.ndim == 3, f"Expected combined features to have 3 dimensions, got {features.ndim}"
        assert features.size(2) == 128, f"Expected feature dimension to be 128, got {features.size(2)}"

        output = self.linear(features.mean(dim=1))
        print(f"Output shape: {output.shape}")
        assert output.ndim == 2, f"Expected output to have 2 dimensions, got {output.ndim}"
        assert output.size(1) == 2, f"Expected output dimension to be 2, got {output.size(1)}"

        return output


# Initialize the model, optimizer, and loss function
model = Terminator()
optimizer = Kar3(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

if args.mode == "train":
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            input_ids = torch.stack(batch["input_ids"])  # Convert input_ids to tensor
            labels = torch.tensor(batch["label"])  # Convert labels to tensor
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = torch.stack(batch["input_ids"])  # Convert input_ids to tensor
                labels = torch.tensor(batch["label"])  # Convert labels to tensor
                outputs = model(input_ids)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.4f}")
else:
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.stack(batch["input_ids"])
            labels = torch.tensor(batch["label"])
            outputs = model(input_ids)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
