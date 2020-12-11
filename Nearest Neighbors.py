import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from time import time

# Model parameters
CLASS_NUM = 2

IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180

data_dir = './dataset'


# Load dataset
Transforms = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(num_output_channels=1)
])

dataset = torchvision.datasets.ImageFolder(
    root=data_dir,
    transform=Transforms
)


t0 = time()
data = []
for (image, label) in dataset:
    x = torch.flatten(image, 1)
    x = x[0].detach().numpy()
    data.append(x)
labels = dataset.targets
print("[INFO] LOADED", len(labels), "IMAGES in %0.3fs" % (time() - t0))



# SPLIT DATA INTO TRAINING AND TESTING SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels,
    test_size=0.20,
    stratify=labels
)



from sklearn.metrics import classification_report
def EVALUATE(clf, clf_name):
    print("[",clf_name,"]")
    print("classification report on the test set")
    t0 = time()
    y_pred = clf.predict(X_test)
    print("done in %0.3fs" % (time() - t0))
    print(classification_report(y_test, y_pred))




from sklearn import neighbors
N_NEIGHBORS = 15
WEIGHTS = 'distance'
nbrs = neighbors.KNeighborsClassifier(N_NEIGHBORS, weights=WEIGHTS, algorithm='kd_tree', metric='euclidean', n_jobs=-1)
nbrs.fit(X_train, y_train)
EVALUATE(nbrs, "Nearest Neighbors")