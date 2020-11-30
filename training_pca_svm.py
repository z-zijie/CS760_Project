import argparse
from imutils import paths
import cv2
import numpy as np
from time import time


PARSER = argparse.ArgumentParser()
PARSER.add_argument("-d", "--dataset", required=True, help="path to dataset")
args = vars(PARSER.parse_args())

# IMAGE SIZE
img_height = 224
img_width = 224

# LOADING IMAGES
print("[INFO] LOADING IMAGES...")
t0 = time()
data_dir = args["dataset"]
PATH = list(paths.list_images(data_dir))

data = []
label = []
for img_path in PATH:
    image = cv2.imread(img_path, 0)
    image = cv2.resize(image, (img_width, img_height))
    
    data.append(image.flatten())
    label.append(img_path.split('\\')[1])
    
    if len(label) % 500 == 0:
        print("[INFO]", len(label), "images have been loaded.") 

data = np.array(data, dtype="float32")
label = np.array(label)
print("[INFO] LOADED", len(label), "IMAGES in %0.3fs" % (time() - t0))


# split data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data, label,
    test_size=0.20,
    stratify=label,
    random_state=42
)


# Compute a PCA (eigenfaces)
from sklearn.decomposition import PCA
n_components = 15
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))
eigenfaces = pca.components_.reshape((n_components, img_height, img_width))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

# Train a SVM classification model
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {
    # 'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'C': [1e3],
    # 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
    'gamma': [0.001]
}
clf = GridSearchCV(
    SVC(kernel='rbf',
        class_weight='balanced',
        probability = True
        ),
    param_grid,
    verbose=100,
    n_jobs=-1
)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# Quantitative evaluation of the model quality on the test set
from sklearn.metrics import classification_report
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
print(classification_report(y_test, y_pred))

# SAVING MODEL
from joblib import dump
dump(clf, 'clf.model')
dump(pca, 'pca.model')
print("[INFO] MODEL SAVED AT clf.model AND pca.model")