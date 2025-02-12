import time

import torch
import torch.nn as nn
from sympy.core.random import randint
from torch.sparse import log_softmax
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch.autograd as autograd


class Generator(nn.Module):
    def __init__(self, num_classes):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, 100)  # Embedding pour les classes
        self.conv1 = torch.nn.utils.spectral_norm(nn.ConvTranspose2d(100 + 100, 112, 4, 1, 0))
        self.batchnorm1 = nn.BatchNorm2d(112)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.ConvTranspose2d(112, 56, 4, 2, 1)
        self.batchnorm2 = nn.BatchNorm2d(56)
        self.relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.ConvTranspose2d(56, 28, 4, 2, 1)
        self.conv4 = nn.ConvTranspose2d(28, 1, 4, 2, 3)
        self.lineaire=nn.Linear(28*28,784)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise, labels):
        label_embedded = self.label_embedding(labels)  # Embedding de la classe
        x = torch.cat((noise, label_embedded.unsqueeze(2).unsqueeze(3)), dim=1)  # Combine le bruit et l'embedding
        x = self.conv1(x)
        # print(x.size(),"conv1")
        x = self.batchnorm1(x)
        # print(x.size(),"batchnorm1")
        x = self.relu1(x)
        # print(x.size(),"relu1")
        x = self.conv2(x)
        # print(x.size(),"conv2")
        x = self.batchnorm2(x)
        # print(x.size(),"batchnorm2")
        x = self.relu2(x)
        # print(x.size(),"relu2")
        x = self.conv3(x)
        # print(x.size(),"conv3")
        x = self.conv4(x)
        # print(x.size(),"conv4")
        x = self.sigmoid(x)

        self.savereseau()
        return x

    def savereseau(self):
        torch.save(self.state_dict(), 'generator_mnist.pth')

    def loadreseau(self):
        self.load_state_dict(torch.load('generator_mnist.pth'))
        self.eval()
    def nb_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Bloc pour classer l'image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(1600, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.fc_real_fake = nn.Linear(512, 1)  # Pour évaluer si l'image est réelle ou fausse

    def forward(self, x):
        # Bloc de convolution et de pooling
        x = nn.LeakyReLU(0.2)(self.conv1(x))
        x = nn.LeakyReLU(0.2)(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = nn.LeakyReLU(0.2)(self.conv3(x))
        x = nn.LeakyReLU(0.2)(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten et couches entièrement connectées
        x = x.view(x.size(0), -1)
        x = nn.LeakyReLU(0.2)(self.fc1(x))
        x = self.dropout3(x)


        # Sorties
        class_output = F.log_softmax(self.fc2(x), dim=1)  # Sortie pour la classification
        real_fake_output = torch.sigmoid(self.fc_real_fake(x))  # Sortie pour vérifier si l'image est réelle
        # print(real_fake_output)
        return class_output, real_fake_output

    def savereseau(self):
        torch.save(self.state_dict(), 'discriminator_mnist.pth')

    def loadreseau(self):
        self.load_state_dict(torch.load('discriminator_mnist.pth'))
        self.eval()
    def nb_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



    def gradient_penalty(self, real_images, fake_images, batch_size):
         # Interpolation between real and fake images
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real_images.device)
        interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
        interpolated_images.requires_grad_(True)

            # Get discriminator output for interpolated images
        _, interpolated_preds = self(interpolated_images)

        # Calculate gradients of the discriminator output with respect to the interpolated images
        gradients = autograd.grad(
            outputs=interpolated_preds,
            inputs=interpolated_images,
            grad_outputs=torch.ones_like(interpolated_preds),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Reshape gradients to calculate the norm
        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)

            # Calculate the gradient penalty
        penalty = ((gradients_norm - 1) ** 2).mean()
        return penalty




def load_cifar10_data(n_samples=None, classes=None):
    if classes is None:
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    X_train = datasets.MNIST(
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


def train(data_loader, epochs=10,resume=False):
    generator = Generator(num_classes=10)
    discriminator = Discriminator()
    print(generator.nb_parameters())
    print(discriminator.nb_parameters())
    if resume:
        generator.loadreseau()
        discriminator.loadreseau()
    generator.train()
    discriminator.train()


    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
    loss_fn_class = nn.CrossEntropyLoss()  # Pour les classes
    loss_fn_real_fake = nn.BCELoss()  # Pour la vérification réelle/faux

    # Liste pour stocker chaque type de perte
    gen_loss_class_list = []
    gen_loss_real_fake_list = []
    disc_loss_class_real_list = []
    disc_loss_class_fake_list = []
    disc_loss_real_or_fake_real_list = []
    disc_loss_real_or_fake_fake_list = []
    real_loss_class=0
    disc_loss_real_or_fake_real=0
    disc_loss_real_or_fake_fake=0
    fake_loss_class=0

    for epoch in range(epochs):
        start_time = time.time()
        for nb, (real_images, real_labels) in enumerate(data_loader):


            batch_size = real_images.size(0)  # Assurez-vous d'utiliser la taille du batch ici
            noise = torch.randn(batch_size, 100, 1, 1)
            torch_labels = torch.tensor(real_labels, dtype=torch.long)

            # Générer des images fake avec le générateur
            fake_images = generator(noise, torch_labels)
            optimizer_disc.zero_grad()
            if nb%1==0:
                # Entraînement du Discriminateur


                real_class_preds, real_fake_preds = discriminator(real_images)
                fake_class_preds, fake_fake_preds = discriminator(fake_images.detach())

                real_loss_class = loss_fn_class(real_class_preds, real_labels)# À quel point l'image réelle est bien classifiée.
                fake_loss_class = loss_fn_class(fake_class_preds, real_labels)# a quelle point l'image fake est bien deviné
                disc_loss_real_or_fake_real = loss_fn_real_fake(real_fake_preds, torch.full((batch_size, 1), 0.9))#a quelle point l'image reel a bien été deviné reel
                disc_loss_real_or_fake_fake = loss_fn_real_fake(fake_fake_preds, torch.full((batch_size, 1), 0.1))# a quelle point l'image fake a bien été deviné fake
                gradient_penalty = discriminator.gradient_penalty(real_images, fake_images, batch_size)
                if epoch < (epochs / 2):
                    disc_loss=disc_loss_real_or_fake_real+disc_loss_real_or_fake_fake+gradient_penalty
                else:
                    disc_loss = real_loss_class + disc_loss_real_or_fake_real + disc_loss_real_or_fake_fake + gradient_penalty+fake_loss_class
                disc_loss.backward(retain_graph=True)
                optimizer_disc.step()

            # Entraînement du Générateur
            optimizer_gen.zero_grad()
            fake_class_preds, fake_fake_preds = discriminator(fake_images)
            gen_loss_class = loss_fn_class(fake_class_preds, real_labels)
            gen_loss_real_fake =loss_fn_real_fake(fake_fake_preds, torch.full((batch_size, 1), 0.9))
            if epoch<(epochs/2):
                gen_loss = gen_loss_real_fake
            else:
                gen_loss = gen_loss_class + gen_loss_real_fake


            gen_loss.backward()
            optimizer_gen.step()

            # Enregistrement des pertes individuelles
            gen_loss_class_list.append(gen_loss_class.item())#
            gen_loss_real_fake_list.append(gen_loss_real_fake.item())
            disc_loss_class_real_list.append(real_loss_class.item())
            disc_loss_class_fake_list.append(fake_loss_class.item())
            disc_loss_real_or_fake_real_list.append(disc_loss_real_or_fake_real.item())
            disc_loss_real_or_fake_fake_list.append(disc_loss_real_or_fake_fake.item())
        end_time = time.time()
        print(f"Epoch [{epoch + 1}/{epochs}], Discriminator Loss: {disc_loss.item()}, Generator Loss: {gen_loss.item()},image par seconde: {len(data_loader.dataset) / (end_time - start_time)}")
        if epoch % 10 == 0:
            generator.savereseau()
            discriminator.savereseau()
            test(generator, num_classes=10)

    # Affichage des courbes des différentes pertes
    plt.figure(figsize=(12, 8))

    # Affichage des pertes sous forme de points
    plt.scatter(range(len(gen_loss_class_list)), gen_loss_class_list, label="Generator Classification Loss", s=10)
    plt.scatter(range(len(gen_loss_real_fake_list)), gen_loss_real_fake_list, label="Generator Real/Fake Loss", s=10)
    plt.scatter(range(len(disc_loss_class_real_list)), disc_loss_class_real_list,
                label="Discriminator Real Classification Loss", s=10)
    plt.scatter(range(len(disc_loss_real_or_fake_real_list)), disc_loss_real_or_fake_real_list,
                label="Discriminator Real Real/Fake Loss", s=10)
    plt.scatter(range(len(disc_loss_real_or_fake_fake_list)), disc_loss_real_or_fake_fake_list,
                label="Discriminator Fake Real/Fake Loss", s=10)

    # Étiquettes et légende
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Sauvegarde des modèles
    generator.savereseau()
    discriminator.savereseau()




def test(generator, num_classes=10):
    generator.eval()
    images_per_class = 1  # Nombre d'images à générer par classe

    # Créer un tableau pour stocker les images générées
    generated_images = []

    # Générer une image pour chaque classe
    for class_label in range(num_classes):
        noise = torch.randn(images_per_class, 100, 1, 1)  # Bruit pour générer une image
        class_tensor = torch.tensor([class_label])
        fake_image = generator(noise, class_tensor)  # Générer l'image
        generated_images.append(fake_image)

    # Convertir la liste en un tenseur

    generated_images = torch.cat(generated_images)

    # Load a sample image from the dataset
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    sample_images = [dataset[i][0] for i in range(num_classes)]

    # Visualize the images
    fig, axes = plt.subplots(2, num_classes, figsize=(num_classes * 3, 6))
    for i in range(num_classes):
        # Display generated images
        img = generated_images[i]
        axes[0, i].imshow(img.detach().cpu().numpy().squeeze())
        axes[0, i].set_title(f'Generated Class {i}')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # Display sample images from the dataset
        sample_img = sample_images[i]
        axes[1, i].imshow(sample_img.numpy().squeeze())
        axes[1, i].set_title(f'Sample Class {i}')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    plt.tight_layout()
    plt.show()


def main(mode="train"):
    batch_size = 100  #nb imaga par batch
    nb_image_by_batch = 6000 #nb image étudier

    epochs = 100
    if mode == "train":
        data = load_cifar10_data(n_samples=nb_image_by_batch)
        data = create_dataloader(data, batch_size=batch_size)

        train(data, epochs=epochs)
    elif mode == "re":
        data = load_cifar10_data(n_samples=nb_image_by_batch)
        data = create_dataloader(data, batch_size=batch_size)
        train(data, epochs=epochs, resume=True)  # Reprendre l'entraînement
    else:
        generator = Generator(num_classes=10)
        generator.loadreseau()
        test(generator, num_classes=10)


if __name__ == "__main__":
    mode = "train"
    torch.manual_seed(time.time())

    main(mode)
