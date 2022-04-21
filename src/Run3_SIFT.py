import os
import re

import cv2
import numpy as np
import sklearn
from scipy.cluster import vq
from skimage.util.shape import view_as_blocks
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.random import sample_without_replacement
from tqdm import tqdm, trange
from yellowbrick.cluster import KElbowVisualizer

# TODO: 借鉴下以下的思路
# https://github1s.com/FrankWJW/SceneRecognition/blob/master/code/cv3/patch_vocabulary.m

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
    scaler.transform(x)
    image = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return image

def readImg(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    # img = normalisation(img) # 归一化 zero mean and unit variance
    img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_NEAREST)
    return img

def img2Kp_Des(Path,):
    desList = []
    kpList = []
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
            img_mat = normalisation(readImg(imgFullPath)) # 读取图片
            sift = cv2.xfeatures2d.SIFT_create()
            # orb = cv2.ORB_create()
            kp, des = sift.detectAndCompute(img_mat,None)
            img_counter += 1
            kpList.append(kp)
            desList.append(des)
            labelVector.append(class_index)
        class_counter[class_index] += img_counter

    return np.array(desList), np.array(labelVector), np.sum(class_counter, dtype=int), class_counter

def list2vstack(desList):# TODO: 太慢了，看看怎么加速
    start = desList[0]
    for des in desList[1:]:
        start = np.vstack((start,des))
    return start

def kMeans(data, n_training_samples=1024,n_clusters=492):
    data = data.reshape(-1,128)
    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=n_training_samples,verbose=1)
    # for _ in trange(epochs):
    #     training_data = sample_without_replacement(data.shape[0], n_training_samples)
    model.fit(data)
    return model, model.cluster_centers_

# def KMeans_clusters_selctor(data):
#     model = MiniBatchKMeans(batch_size=2048 ,verbose=1)
#     # k is range of number of clusters.
#     visualizer = KElbowVisualizer(model, k=(490,510), timings= True) 
#     # visualizer = KElbowVisualizer(model, k=(2,30),metric='silhouette', timings= True)
#     visualizer.fit(data)        # Fit the data to the visualizer
#     visualizer.show() 
    
def img2Hist(model, desList, image_counter, no_clusters,):
    img_hists = np.array([np.zeros(no_clusters) for _ in range(image_counter)])
    for i in trange(image_counter, desc='extracting feature per image'):
        feature = np.array(desList[i])
        feature = feature.reshape(-1, 128)
        idx = model.predict(feature)
        img_hists[i][idx] += 1
        # vq
        # predict_idies, distance = vq(feature, model.cluster_centers_)
        # for dix in predict_idies:
        #     img_hists[i][idx] += 1/len(predict_idies) # 归一化

    return img_hists

# https://blog.csdn.net/qq_36622009/article/details/102895411
def idf_and_norm(img_histogram_list):
    words_occurrences = np.sum((img_histogram_list > 0) * 1, axis=0)  # 求出每个word在所有图片中出现的频数
    idf = np.array(np.log((1.0 * len(img_histogram_list) + 1) / (1.0 * words_occurrences + 
         1)),'float32')     
 
    img_book_list = img_histogram_list * idf
    img_book_list = sklearn.preprocessing.normalize(img_book_list, norm='l2')  # 归一化
    return idf, img_book_list


# def svcParamSelection(X, y, nfolds):
#     # best : 0.1, 0.5
#     Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
#     gammas = [0.1, 0.11, 0.095, 0.105]
#     param_grid = {'C': Cs, 'gamma' : gammas}
#     grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds, n_jobs=4, verbose=1)
#     grid_search.fit(X, y)
#     grid_search.best_params_
#     return grid_search.best_params_

# def findSVM(im_features, train_labels):
#     features = im_features
#     params = svcParamSelection(features, train_labels, 5)
#     C_param, gamma_param = params.get("C"), params.get("gamma")
#     print(C_param, gamma_param)
  
#     svm = SVC(C =  C_param, gamma = gamma_param,)
#     svm.fit(features, train_labels)
#     return svm

def OvRLCs(data, label, n_models):

    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.9, shuffle=True)
    estimators = []
    for index in range(n_models):
        # model = ('svc_'+str(index), SVC(C=0.5, gamma=0.1))
        model = ('lr_'+str(index), LinearSVC(multi_class='ovr'))
        # model = ('lr_'+str(index), LogisticRegression())
        estimators.append(model)
        # model = ('svc_'+str(index), LinearSVC(multi_class='ovr'))
        # estimators.append(model)

    clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=5,
        verbose=1,
        n_jobs=4,
    )

    # clf = BaggingClassifier(
    #     # estimators=estimators, 
    #     # final_estimator=LinearSVC(),
    #     n_estimators=n_models,
    #     # cv=5,
    #     verbose=1,
    #     n_jobs=4,
    # )
    clf.fit(X_train, y_train)
    # clf = findSVM(X_train, y_train)

    return clf, metrics.classification_report(y_test,clf.predict(X_test), target_names=labels)

np.random.seed(100) 

img_size = 128 #越小获取到的信息可能更多
n_clusters = 500
n_models = 15
n_training_samples = 1024 # batch_size


# np.random.seed(48) 
# 0.19
# img_size = 128 #越小获取到的信息可能更多
# n_clusters = 200
# n_models = 15 
# n_training_samples = 1024 # batch_size
# step_size = 3

# 0.30 all data
# img_size = 128
# n_clusters = 200
# n_models = 20 
# n_training_samples = 2048
# step_size=2

# 0.29
# img_size = 128
# n_clusters = 100
# n_models = 20 
# n_training_samples = 1024
# step_size=2

imgVector_train, labelVector_train, img_counter, class_counter = img2Kp_Des(trainingDatasetPath, )
imgVector_train = list2vstack(imgVector_train)
print(imgVector_train.shape) # (759487, 128)
print('Training KMeans...')
kmeans, visual_words = kMeans(imgVector_train, n_training_samples=n_training_samples, n_clusters=n_clusters)
# KMeans_clusters_selctor(imgVector_train)
print('Extracting features...')
_, imgFeature_train = idf_and_norm(img2Hist(kmeans, imgVector_train, img_counter, n_clusters))
# idf, imgFeature_train = idf_and_norm(imgFeature_train)
# imgVector_train = normalisation(imgFeature_train)
# print(imgFeature_train.shape) #(1500, 500)
print('Training OvRLCs...')
final_model, score = OvRLCs(imgFeature_train, labelVector_train, n_models=n_models)
print(score)
