import numpy as np
import fire
import data_processing as dp
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

class basic_block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


'''
    Defines the network architecture, activations and regularizers.
    Forward prop.
'''
class resnet(nn.Module):
    def __init__(self, 
                 block, 
                 num_blocks, 
                 num_classes=10):
        super(resnet, self).__init__()
        self.in_planes = 32
        # First convolutional layer must have the kernel size 5x5,
        # stride 1 and the total number of kernels 32, according to the 
        # project assignment: 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, 
                               stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.activation = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 256, num_blocks[3], stride=2)        
        self.dense = nn.Linear(256*block.expansion, num_classes)

    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


def ResNet18():
    return resnet(basic_block, [2,2,2,2])

'''
    Sets the loss and optimization criterion and number of epochs.
    They were chosen heuristically.
'''
def set_optimization(model):
    # This criterion combines nn.LogSoftmax() and nn.NLLLoss() 
    # in one single clas
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, 
                          weight_decay=2e-4)
    epochs = 10
    return criterion, optimizer, epochs

'''
    forward + backward prop for 1 epoch
    prints the loss for every minibatch (2000 images)
'''
def train_model(model, trainloader, criterion, optimizer, epoch, device):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[epoch: %d, batch: %5d] loss: %.3f' %
                   (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

'''
    Tests the model accuracy over the test data in one epoch
    Prints the average loss
'''
def test_model(model, testloader, epoch, device):
    correct, total = 0, 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%\n' % (
          100 * correct / total))

'''
    Saves the model to the directory Model
'''
def save_model(net):
    torch.save(net.state_dict(), f="Model/model.model")
    print("Model saved successfully.")

'''
    Loads the pretrained network. 
'''
def load_model(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        net.load_state_dict(torch.load("Model/model.model", 
                            map_location='gpu')) 
    except RuntimeError:
        print("Runtime Error!")
        print(("Saved model must have the same network architecture with"
               " the CopyModel.\nRe-train and save again or fix the" 
               " architecture of CopyModel."))
        exit(1) # stop execution with error

'''
    Trains network using GPU, if available. Otherwise uses CPU.
'''
def set_device(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on: %s\n" %device)
    # .double() will make sure that  MLP will process tensor
    # of type torch.DoubleTensor:
    return net.to(device), device

'''
    Applies the train_model and test_model functions at each epoch
'''
def train():
    # This loads the dataset and partitions it into batches:
    trainset, testset = dp.load_cifar10()
    trainloader, testloader = dp.batch_data(trainset, testset)
    # Loads the model and the training/testing functions:
    net = ResNet18()
    net, device = set_device(net)
    criterion, optimizer, epochs = set_optimization(net)
    
    # Print the train and test accuracy after every epoch:
    for epoch in range(epochs):
        train_model(net, trainloader, criterion, optimizer, epoch, device)
        test_model(net, testloader, epoch, device)

    print('Finished Training')   
    # Save the model:
    save_model(net)

'''
    Classifies the image whose path entered on the terminal.
'''
def test(image_path):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    img_tensor = dp.load_test_image(image_path).unsqueeze(0)
    net = ResNet18()
    load_model(net)
    outputs = net(img_tensor)
    _, predicted = torch.max(outputs.data, 1)
    print("Predicted: %s" %classes[predicted[0]])


if __name__ == "__main__":
    fire.Fire()

