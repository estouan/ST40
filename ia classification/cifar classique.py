import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Conv2d, Linear, Module, functional as F, NLLLoss, MaxPool2d
from torch import no_grad
import torch.optim as optim
from torch.nn import Dropout
from PIL import Image
import os

def denormalize(tensor):
    # Re-inverser la normalisation
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    tensor = tensor * std[:, None, None] + mean[:, None, None]  # dénormaliser
    return tensor

def load_cifar10_data(n_samples, classes):

    X_train = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )
    idx = np.isin(X_train.targets, classes)
    X_train.data = X_train.data[idx][:n_samples]
    X_train.targets = np.array(X_train.targets)[idx][:n_samples]

    return X_train


def create_dataloader(dataset, batch_size=1):
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        targets = torch.tensor(targets, dtype=torch.long)  # Ensure targets are of type Long
        return images, targets

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def show_samples(model, data_loader, n_samples_show=6):
    model.eval()
    data_iter = iter(data_loader)
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(15, 3))
    samples_shown = 0

    with torch.no_grad():
        while samples_shown < n_samples_show:
            images, targets = data_iter.__next__()
            outputs = model(images)
            preds = outputs.argmax(dim=1, keepdim=True)

            for i in range(len(images)):
                if samples_shown >= n_samples_show:
                    break
                axes[samples_shown].imshow(images[i].cpu().permute(1, 2, 0).numpy())  # Transfert sur CPU pour affichage
                axes[samples_shown].set_xticks([])
                axes[samples_shown].set_yticks([])
                axes[samples_shown].set_title(f"Pred: {preds[i].item()}, Label: {targets[i].item()}")
                samples_shown += 1

    plt.show()





class ClassiqueNet(Module):
    def __init__(self,nbclass,  dropout=0.3):
        super(ClassiqueNet, self).__init__()

        # CONV => RELU => CONV => RELU => POOL => DROPOUT (Block 1)
        self.conv1 = Conv2d(3, 32, kernel_size=3, padding='same')
        self.conv2 = Conv2d(32, 32, kernel_size=3)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = Dropout(0.25)

        # CONV => RELU => CONV => RELU => POOL => DROPOUT (Block 2)
        self.conv3 = Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv4 = Conv2d(64, 64, kernel_size=3)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = Dropout(0.25)

        # FLATTERN => DENSE => RELU => DROPOUT
        self.fc1 = Linear(2304, 512)  # Assuming input image size of 32x32
        self.dropout3 = Dropout(0.5)
        self.fc2 = Linear(512, 128)  # Adjusted for the number of qubits
        self.fc3 = Linear(128,nbclass )

        print(" parameter =",sum(p.numel() for p in self.parameters()if p.requires_grad))

        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten and Fully Connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.fc3(x)



        # Output softmax for classification
        return self.log_softmax(x)  # Log softmax for 10 classes


def train_model(model, train_loader, epochs=10, lr=0.001, weight_decay=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_func = torch.nn.NLLLoss()
    loss_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        scheduler.step()
        loss_list.append(sum(total_loss) / len(total_loss))
        print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))

    plt.plot(loss_list)
    plt.title("Hybrid NN Training Convergence")
    plt.xlabel("Training Iterations")
    plt.ylabel("Neg. Log Likelihood Loss")
    plt.show()
    torch.save(model.state_dict(), "modelclassique.pt")


def test_model(model, test_loader):
    model.eval()
    loss_func = NLLLoss()
    total_loss = []
    correct = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = loss_func(output, target)
            total_loss.append(loss.item())

    print(
        "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
            sum(total_loss) / len(total_loss), correct / len(test_loader) / test_loader.batch_size * 100
        )
    )


def predict_image(model, image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the model and image to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Make the prediction
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True).item()

    return prediction


def test_perso(model, directory_path="picture"):
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    images = []
    predictions = []

    # Iterate over all images in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)  # Preprocess and add batch dimension

            # Make the prediction
            with torch.no_grad():
                output = model(image)
                prediction = output.argmax(dim=1, keepdim=True).item()

            images.append(image.cpu().squeeze(0))  # Remove batch dimension and move to CPU
            predictions.append(prediction)

    # Display the images with their predicted labels
    n_samples_show = len(images)
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(15, 3))

    if n_samples_show == 1:
        axes = [axes]  # Ensure axes is always a list

    for i in range(len(images)):
        if i >= n_samples_show:
            break
        image = denormalize(images[i].cpu()).permute(1, 2, 0).numpy()  # Dé-normalisation + permutation pour imshow
        axes[i].imshow(image)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"Pred: {predictions[i]}")
        i += 1
    plt.show()


def main(mode='test'):
    batch_size = 10
    n_samples_train = 1000
    n_samples_test = 1000
    epoch=50
    classe=list(range(3))
    nbclasse=len(classe)


    X_train = load_cifar10_data(n_samples_train ,classes=classe)
    train_loader = create_dataloader(X_train, batch_size=batch_size)
    X_test = load_cifar10_data(n_samples_test,classes=classe)
    test_loader = create_dataloader(X_test, batch_size=batch_size)


    model = ClassiqueNet(nbclasse)

    if mode == 'train':
        train_model(model, train_loader, epochs=epoch)
        show_samples(model, test_loader)
    elif mode == 'test':
        model.load_state_dict(torch.load("modelclassique.pt"))
        test_model(model, test_loader)
        show_samples(model, test_loader)
    elif mode == "perso":
        model.load_state_dict(torch.load("modelclassique.pt"))
        directory_path = "picture"
        test_perso(model=model, directory_path=directory_path)



if __name__ == "__main__":
    main(mode='test')
