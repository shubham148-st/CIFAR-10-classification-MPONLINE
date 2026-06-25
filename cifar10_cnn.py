import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = 128

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

 # two layer CNN
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            # 1st layer
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)
            #2nd layer
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)

            # Fully connected layers
            # Images start at 32x32. Two 2x2 pools reduce it to 8x8.
            self.fc1 = nn.Linear(64 * 8 * 8, 512)
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 10)
            
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool1(self.relu(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu(self.bn2(self.conv2(x))))
            
            x = torch.flatten(x, 1)
            
            x = self.relu(self.fc1(x))
            x = self.dropout1(x)
            x = self.fc2(x)
            return x

    net = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001) #used Adam for fater conv

    print("Starting training...")
    num_epochs = 20
    epoch_accuracies = []

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f'[{epoch + 1}/{num_epochs}] training loss: {running_loss / len(trainloader):.3f}')

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        epoch_accuracies.append(epoch_acc)
        print(f'Accuracy after epoch {epoch + 1}: {epoch_acc:.2f} %')

    print('Finished Training')

    # Plot the accuracy graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_accuracies, marker='o', linestyle='-', color='g')
    plt.title('Test Accuracy over Epochs ')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.xticks(range(1, num_epochs + 1))
    plt.savefig('accuracy_graph.png')
    print("Accuracy graph saved to accuracy_graph.png")

if __name__ == '__main__':
    main()
