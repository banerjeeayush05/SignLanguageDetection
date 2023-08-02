import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

#Defining Constants
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.01

#Creating model class
class HandSignTinyVGGModel(nn.Module):
    #Copying architecture from the TinyVGG Model
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        #Creating first convolutional block
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        #Creating second convolutional block
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        #Creating classifying block
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16,
                      out_features=output_shape)
        )
    #Defining Forward Layer
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

#Train one epoch method
def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        #Calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        #Backpropogate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")

#Train across multiple epochs
def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("-----------------------")
    print("Training complete")

#Main Script
if(__name__ == '__main__'):
    transform = transforms.Compose([transforms.Resize(size = (64, 64)), transforms.ToTensor()])

    #Create and organize data
    train_data = datasets.ImageFolder(root = "Data", transform = transform)
    train_data_loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, num_workers = 0, shuffle = True)

    #Setting up device (cpu or gpu)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")
    hand_signals_net = HandSignTinyVGGModel(input_shape = 3, hidden_units = 10, output_shape = 3).to(device)

    #Establishing loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(hand_signals_net.parameters(), lr = LEARNING_RATE)

    #Implement train method
    torch.manual_seed(42)
    train(hand_signals_net, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    #Save the trained model
    torch.save(hand_signals_net.state_dict(), "hand_sign_tiny_vgg_model.pth")
    print("Model trained and stored at feed_forward_net.pth")