import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np

writer = SummaryWriter('runs/Mask')
# Model parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCH = 10
CLASS_NUM = 2

IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180

data_dir = './dataset'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Load dataset
Transforms = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomApply(torch.nn.ModuleList([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(degrees=15),
    ]), p=0.3),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.ImageFolder(
    root=data_dir,
    transform=Transforms
)
classes = ('with_mask', 'without_mask')



# Split dataset
from sklearn.model_selection import train_test_split
dataset_size = len(dataset)
testing_split = 0.2
train_idx, test_idx = train_test_split(range(dataset_size),test_size=testing_split )

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

trainloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler
)
testloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=test_sampler
)



# Show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.axis('off')
#     fig1 = plt.figure(figsize = (32,4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
img_grid = torchvision.utils.make_grid(images, nrow=16)
imshow(img_grid)



def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))






# write to tensorboard
matplotlib_imshow(img_grid, one_channel=True)
writer.add_image('mask_images', img_grid)



import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),
                     
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),
            
            nn.Dropout(p=0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16 * 45 * 45, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits


net = Net()
net.to(DEVICE)
net.eval()


def images_to_probs(net, images):
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]



def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig



import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


running_loss = 0.0
for epoch in range(EPOCH):  # loop over the dataset multiple times

    loop = tqdm(enumerate(trainloader), total=len(trainloader))
    i = 0
    for batch_idx, (inputs, targets) in loop:
        # get the inputs; data is a list of [data, targets]
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # update progress bar
        loop.set_description(f"Epoch [{epoch+1}/{EPOCH}]")
        loop.set_postfix(loss=loss.item())
        
        i += 1
        running_loss += loss.item()
        if i == 85:
            # ...log the running loss
            writer.add_scalar('training loss',
                              running_loss / 85,
                              epoch * len(trainloader) + i)
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(net, inputs, targets),
                              global_step=epoch * len(trainloader) + i)
            running_loss = 0.0

print('Finished Training')



PATH = './mask_net.pth'
torch.save(net.state_dict(), PATH)
writer.add_graph(net, images)
writer.close()



dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))



# Load Net
net = Net()
net.load_state_dict(torch.load(PATH))


# Split dataset
from sklearn.model_selection import train_test_split
dataset_size = len(dataset)
testing_split = 0.2
train_idx, test_idx = train_test_split(range(dataset_size),test_size=testing_split )

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

trainloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler
)
testloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=test_sampler
)


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images.to(DEVICE)
        labels.to(DEVICE)
        
        logits = net(images)
        probas = F.softmax(logits, dim=1)
        _, predicted = torch.max(probas, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))



class_correct = list(0. for i in range(CLASS_NUM))
class_total = list(0. for i in range(CLASS_NUM))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images.to(DEVICE)
        labels.to(DEVICE)
        
        logits = net(images)
        probas = F.softmax(logits, dim=1)
        _, predicted = torch.max(probas, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(CLASS_NUM):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))