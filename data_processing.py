import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
# Image class comes from a package called pillow 
# PIL used as the format for passing images into torchvision

def chained_transformation():
    preprocess = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), 
                                     (0.5, 0.5, 0.5))])
    return preprocess

def load_cifar10():
    preprocess = chained_transformation()
    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True,
                                            download=True, 
                                            transform=preprocess)
    testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False,
                                           download=True, 
                                           transform=preprocess)
    return trainset, testset

def load_test_image(image_path):
    image = Image.open(image_path)
    image.show()
    preprocess = chained_transformation()
    img_tensor = preprocess(image)
    return img_tensor

def batch_data(trainset, testset, batch_size=4):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

def test_loading(trainloader, classes):
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # show images
    imshow(torchvision.utils.make_grid(images))

def main():
    trainset, testset = load_cifar10()
    trainloader, testloader = batch_data(trainset, testset)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    test_loading(trainloader, classes)


if __name__ == '__main__':
    main()



