import os
import re

import cv2
import numpy as np
import scipy as sp
from skimage.util.shape import view_as_blocks
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.random import sample_without_replacement
from tqdm import tqdm, trange

np.random.seed(42) 

trainingDatasetPath = 'data/training'
testDatasetPath = 'data/testing'

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


def normalisation(x):  
    scaler = StandardScaler().fit(x)
    return scaler.transform(x)

def readImg(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    # img = normalisation(img) # 归一化 zero mean and unit variance
    img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_NEAREST)
    return img

def split_patchs(img, patch_size=8):
    patches = []
    if img.shape != (img_size,img_size):
        img = cv2.resize(img,(img_size,img_size))

    for block in np.reshape(view_as_blocks(img, block_shape=(patch_size, patch_size)), (-1, patch_size, patch_size)): # 将图片分割为8x8的窗口
        patches.append(np.reshape(block[::4,::4], (-1,))) # 每隔4个采样，拉直

    return np.array(patches)

def img2vectors(Path):
    imgVector = []
    labelVector = []
    class_counter = np.zeros((15, ), dtype=int)
    for dirName in tqdm(os.listdir(Path), desc='reading images...'):
        if(dirName.startswith('.')): continue # Ignore the .DS_Stroe
        dirFullPath = os.path.join(Path, dirName)
        img_counter = 0
        class_index = labels[dirName]
        for imgPath in os.listdir(dirFullPath):
            if(imgPath.startswith('.')): continue # Ignore the .DS_Stroe
            imgFullPath = os.path.join(dirFullPath, imgPath)
            img_vectors = normalisation(split_patchs(readImg(imgFullPath))) # 每张图片分割为 1024, (16, )的vector
            img_counter += 1
            imgVector.append(img_vectors)
            labelVector.append(class_index)
        class_counter[class_index] += img_counter
    return np.array(imgVector), np.array(labelVector), np.sum(class_counter, dtype=int), class_counter

def kMeans(data, n_training_samples=1024,n_clusters=500):

    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=n_training_samples,verbose=1)
    # for _ in trange(epochs):
    #     training_data = sample_without_replacement(data.shape[0], n_training_samples)
    model.fit(data)
    return model, model.cluster_centers_
    
def img2Feature(model, imgVector_train, image_counter, class_counter ,no_clusters,):
    im_features = np.array([np.zeros(no_clusters) for _ in range(image_counter)])
    for i in trange(len(class_counter), desc='extracting feature per class'):
        for j in range(class_counter[i]):
            feature = imgVector_train[j]
            feature = feature.reshape(-1, 4)
            idx = model.predict(feature)
            im_features[i][idx] += 1

    return im_features

def OvRLCs(data, label):

    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.9)

    estimators = [
        ('lr_1', LogisticRegression(multi_class='ovr')),
        ('lr_2', LogisticRegression(multi_class='ovr')),
        ('lr_3', LogisticRegression(multi_class='ovr')),
        # ('svr_1', LinearSVC(multi_class='ovr')),     
        # ('svr_2', LinearSVC(multi_class='ovr')),
        # ('svr_3', LinearSVC(multi_class='ovr')),
    ]
    clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=5,
        verbose=1,
        n_jobs=4,
    )
    clf.fit(X_train, y_train)

    return clf, clf.score(X_test, y_test)

img_size = 256
n_clusters = 500

imgVector_train, labelVector_train, img_counter, class_counter = img2vectors(trainingDatasetPath)
imgVector_train = normalisation(np.reshape(imgVector_train, (-1,4)))
# print(imgFeature_train.shape) # (1536000, 4)
print('Training KMeans...')
kmeans, visual_words = kMeans(imgVector_train, n_training_samples=1024, n_clusters=n_clusters)
print('Extracting features...')
imgFeature_train = img2Feature(kmeans, imgVector_train, img_counter, class_counter, n_clusters)
# imgVector_train = normalisation(imgFeature_train)
# print(imgFeature_train.shape) #(1500, 500)
final_model, score = OvRLCs(imgFeature_train, labelVector_train)
print(score)
