
import os
from turtle import screensize

import cv2
import numpy as np
from sklearn import metrics
from sklearn.model_selection import (ShuffleSplit, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier

labels = {
    'Forest':0, 
    'bedroom':1, 
    'Office':2, 
    'Highway':3, 
    'Coast':4, 
    'Insidecity':5, 
    'TallBuilding':6,
    'industrial':7,
    'Street':8, 
    'livingroom':9,
    'Suburb':10, 
    'Mountain':11, 
    'kitchen':12, 
    'OpenCountry':13, 
    'store':14
    }

trainingDatasetPath = 'data/training'
testDatasetPath = 'data/testing'


def readImg(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    new_length = min(img.shape[0], img.shape[1])
    # crop
    center_x = img.shape[0] // 2
    center_y = img.shape[1] // 2
    x = center_x - new_length//2
    y = center_y - new_length//2
    img = img[y:y+new_length, x:x+new_length]
    img = cv2.resize(img,(16,16),interpolation=cv2.INTER_NEAREST)
    return img

def img2matrix(Path):
    imgFeature = []
    labelVector = []
    for dirName in os.listdir(Path):
        if(dirName.startswith('.')): continue # Ignore the .DS_Stroe
        dirFullPath = os.path.join(Path, dirName)
        
        for imgPath in os.listdir(dirFullPath):
            if(imgPath.startswith('.')): continue # Ignore the .DS_Stroe
            imgFullPath = os.path.join(dirFullPath, imgPath)

            # reshape the matrix to vector
            img_flat = np.reshape(readImg(imgFullPath), (-1,))
            imgFeature.append(img_flat)
            labelVector.append(labels[dirName])
    
    return np.array(imgFeature), np.array(labelVector)

def train(dataset, labels):
    clf = KNeighborsClassifier()
    # clf.fit(x_train, y_train)
    # y_predicted = clf.predict(x_test)
    # accuracy = np.mean(y_test == y_predicted) * 100
    # print('Acc: {0: .1f}%'.format(accuracy))

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    scores = cross_validate(clf, dataset, labels, cv=cv,return_estimator=True)
    clfs = scores['estimator']
    scores.pop('estimator')
    return clfs, scores

def test(groudTruth, predicted):
    return metrics.classification_report(groudTruth, predicted, target_names=labels)

# load the data
imgFeature, labelVector = img2matrix(trainingDatasetPath)
x_train, x_test, y_train, y_test = train_test_split(imgFeature, labelVector, train_size=0.8, random_state=42)
clfs, scores = train(x_train, y_train)
print(scores)

for clf in clfs:
    testAcc = test(y_test, clf.predict(x_test),)
    print(testAcc)
