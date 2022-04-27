
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
    length = min(img.shape[0], img.shape[1])

    x = img.shape[1] // 2 - length//2
    y = img.shape[0] // 2 - length//2

    img = img[y:y+length, x:x+length]
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

def train(imgFeature_train, labelVector_train, n_neighbors):
    X_train, X_test, y_train, y_test = train_test_split(imgFeature_train, labelVector_train, train_size=0.9, shuffle=True)
    # clf = KNeighborsClassifier()
    # clf.fit(x_train, y_train)
    # y_predicted = clf.predict(x_val)
    # accuracy = np.mean(y_val == y_predicted) * 100
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    # accuracy = np.mean(labelVector_train == y_predicted) * 100
    accuracy = metrics.accuracy_score(y_predicted,y_test)
    # print(accuracy) # 30.8
    # print(metrics.classification_report(y_predicted, y_test, target_names=labels))
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
np.random.seed(42)
# n_neighbors=5 # 25 Acc
# clf, accuracy = train(imgFeature_train, labelVector_train, n_neighbors)

# test(testDatasetPath, clf)

# print(len(os.listdir(testDatasetPath))) # 总共2985个测试样本

import tqdm

acc = []
for n_neighbours in tqdm.tqdm(np.arange(start=1,stop=100)):
    clf, accuracy = train(imgFeature_train, labelVector_train, n_neighbours)
    acc.append(accuracy)
import matplotlib.pyplot as plt

plt.plot(np.arange(1,100), acc)
plt.show()
