
import os
import re

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

def train(imgFeature_train, labelVector_train):
    # clf = KNeighborsClassifier()
    # clf.fit(x_train, y_train)
    # y_predicted = clf.predict(x_val)
    # accuracy = np.mean(y_val == y_predicted) * 100
    clf = KNeighborsClassifier()
    clf.fit(imgFeature_train, labelVector_train)
    y_predicted = clf.predict(imgFeature_train)
    accuracy = np.mean(labelVector_train == y_predicted) * 100
    print(accuracy) # 30.8
    print(metrics.classification_report(labelVector_train, y_predicted, target_names=labels))
    return clf, accuracy

def Val(groudTruth, predicted):
    return metrics.classification_report(groudTruth, predicted, target_names=labels)

def test(Path, clf):
    results = []
    fileNames = os.listdir(Path)
    fileNames.sort(key= lambda x:int(x[:-4]))
    for imgPath in fileNames:
        if(imgPath.startswith('.')): continue # Ignore the .DS_Stroe
        imgFullPath = os.path.join(Path, imgPath)

        # reshape the matrix to vector
        img_flat = np.reshape(readImg(imgFullPath), (1, -1))
        y_predicted = clf.predict(img_flat)
        results.append(imgPath + ' ' + str(list(labels.keys())[list(labels.values()).index(y_predicted[0])]))
    
    f=open("results_run_1.txt","w")
    
    f.writelines('\n'.join(results))
    f.close()
    print('Done')


# load the data
imgFeature_train, labelVector_train = img2matrix(trainingDatasetPath)
# x_train, x_val, y_train, y_val = train_test_split(imgFeature, labelVector, train_size=0.8, random_state=42)
# clf, scores = train(x_train, x_val, y_train, y_val)
# print(scores)
clf, accuracy = train(imgFeature_train, labelVector_train)

test(testDatasetPath, clf)

# print(len(os.listdir(testDatasetPath))) # 总共2985个测试样本


