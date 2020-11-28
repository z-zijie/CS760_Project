import joblib
clf = joblib.load('clf.model')

import sys
if len(sys.argv) < 2:
    sys.exit("Need image location.")


import numpy as np
import cv2
img_path = sys.argv[1:]
width, height = 40, 40

# import dnn_model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

def detect_img(net, image):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()
    return detections

def show_detections(image, detections, dim, clf):
    h, w, c = image.shape
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            crop_img = image[startY:endY, startX:endX]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
            resized = resized.astype('float32')
            resized = resized.flatten()
            predict = clf.predict([resized])
            predict_proba = clf.predict_proba([resized])
            confidence = predict_proba[0][predict][0]
            
            text = str(predict==1) + "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            if predict[0]==1:
                cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 1)
                cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 1)
                cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return image

import matplotlib.pyplot as plt

for filename in img_path:
    width, height = 40, 40
    dim = (width, height)
    img = cv2.imread(filename)
    detections = detect_img(net, img)
    img = show_detections(img, detections, dim, clf)
    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # dim = (width, height)
    # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # resized = resized.astype('float32')
    # img = resized.flatten()
    # predict = clf.predict([img])
    # predict_proba = clf.predict_proba([img])
    # print(predict == 1, predict_proba[0][predict], filename)