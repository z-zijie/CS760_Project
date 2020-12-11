# Face Mask Detection(CS760_Project)

## Abstract
In this project, Neural Network, Random-Forest and K-nearest neighbor algorithms were implemented to perform the task of mask detection. Use convolutional neural network(CNN) to extract features of images, and then use fully connected(FC) network for classification. After 100 epochs, the accuracy on the test set reached 96%. The classifier obtained in this project can perform real-time prediction in images, video clips, and network video streams.

## requirements
```
argparse == 1.1
imutils == 0.5.3
opencv-python >= 4.4.0
numpy == 1.18
scikit-learn == 0.23.1
torch===1.7.1
torchvision===0.8.2
torchaudio===0.7.2
tqdm
```

## Dataset
The dataset contains 3400 images, which belong to two categories:
- with mask: __1470__
- without mask: __1930__

These data comes from the following sources:
- Real-World Masked Face Dataset, RMFD
- CASIA-WebFace-Dataset

## Installation
1. clone the repo
   ```
   git@github.com:z-zijie/CS760_Project.git
   ```
2. cd to the cloned repo and create a Python virtual environment
   ```
   mkvirtualenv mask
   ```
3. install the libraries required
   ```
   pip3 install -r requirements.txt
   ```

## Run
1. Random Forest Algorithm
   ```
   python "Random Forest.py"
   ```
2. K Nearest Neighbor Algorithm
   ```
   python "Nearest Neighbors.py"
   ```
3. Neural Networks
   ```
   python "Neural Networks.py"
   ```

## Result
```

[ Random Forest ]
classification report on the test set
done in 0.061s
              precision    recall  f1-score   support

           0       0.92      0.84      0.88       294
           1       0.88      0.95      0.91       386

    accuracy                           0.90       680
   macro avg       0.90      0.89      0.90       680
weighted avg       0.90      0.90      0.90       680

[ Nearest Neighbors ]
classification report on the test set
done in 22.212s
              precision    recall  f1-score   support

           0       0.92      0.73      0.81       294
           1       0.82      0.95      0.88       386

    accuracy                           0.86       680
   macro avg       0.87      0.84      0.85       680
weighted avg       0.87      0.86      0.85       680

[ Neural Net
Accuracy of the network on the test images: 96 %
Accuracy of with_mask : 100 %
Accuracy of without_mask : 97 %
```
![](/Paper/trainingloss.svg)
For more details, please see in the [paper](/Face%20Mask%20Detection.pdf)