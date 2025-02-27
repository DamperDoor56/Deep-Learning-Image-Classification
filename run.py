import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
# Visualization tools
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# Load both the `train` and `valid` datasets for MNIST
train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True)

x_0, y_0 = train_set[0]
#print(x_0)
#print(type(x_0))

# Transform into Tensor
trans = transforms.Compose([transforms.ToTensor()])
x_0_tensor = trans(x_0)
# Check type
#print(x_0_tensor.dtype)
#print(x_0_tensor.min())
#print(x_0_tensor.max())
#print(x_0_tensor.size())

#print(x_0_tensor)

#print(x_0_tensor.device)
# Move to GPU
model = x_0_tensor.to(torch.device("cpu")) 
image = F.to_pil_image(x_0_tensor)
plt.imshow(image, cmap='gray')

## Preparing the data for training: Transforms
trans = transforms.Compose([transforms.ToTensor()])
train_set.transform = trans
valid_set.transform = trans

batch_size = 32

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)

# Creating the model
layers = []
layers

test_matrix = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
)
test_matrix
nn.Flatten()(test_matrix)

batch_test_matrix = test_matrix[None, :]
batch_test_matrix

nn.Flatten()(batch_test_matrix)

nn.Flatten()(test_matrix[:, None])

layers = [
    nn.Flatten()
]
layers

input_size = 1 * 28 * 28

layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
]
layers

layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
    nn.Linear(512, 512),  # Hidden
    nn.ReLU()  # Activation for hidden
]
layers

n_classes = 10

layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
    nn.Linear(512, 512),  # Hidden
    nn.ReLU(),  # Activation for hidden
    nn.Linear(512, n_classes)  # Output
]
layers

model = nn.Sequential(*layers)
model
model.to(device)

next(model.parameters()).device

model = torch.compile(model)
#print(model)

# Training Model
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)

# Calculate the accuracy for each batch
def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

# Train Function to train our model based on the training data
def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

# Validation Function
def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

# Loop
epochs = 5

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()

prediction = model(x_0_tensor)
prediction

prediction.argmax(dim=1, keepdim=True)
print(prediction.argmax(dim=1, keepdim=True))
y_0
print(y_0)