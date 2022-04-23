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
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
# https://github.com/beyzacevik/Scene-Recognition-using-SIFT  Ref

# 读图片（readImg） ——> img2Kp_Des(特征提取) -> kmeans (得到词汇本 vocabulary) -> 把图像特征转换成histograms(img2Hist) -> 训练(svnClf)

def histNorm(hist):
    scaler = MinMaxScaler().fit(hist)
    hist = scaler.transform(hist)
    return hist

def normalisation(x):  
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    image = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return image

def readImg(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = normalisation(img) # 归一化 zero mean and unit variance
    # img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_NEAREST)
    return img

# 特征提取
def img2Kp_Des(Path,):
    desDict = {}
    labelVector = []

    # class_counter = np.zeros((15, ), dtype=int)
    img_counter = 0 
    for dirName in tqdm(os.listdir(Path), desc='reading images...'):
        if(dirName.startswith('.')): continue # Ignore the .DS_Stroe
        dirFullPath = os.path.join(Path, dirName)

        class_index = labels[dirName]

        for imgPath in os.listdir(dirFullPath):
            if(imgPath.startswith('.')): continue # Ignore the .DS_Stroe
            imgFullPath = os.path.join(dirFullPath, imgPath)
            img_mat = readImg(imgFullPath) # 读取图片

            sift = cv2.xfeatures2d.SIFT_create() # (xxx, 128)
            # orb = cv2.ORB_create() # (xxx, 32)
            kp, des = sift.detectAndCompute(img_mat,None)

            desDict[img_counter] = des
            labelVector.append(class_index)
            img_counter += 1
        # class_counter[class_index] += img_counter

    return desDict, list2vstack(labelVector), img_counter

def list2vstack(desList):# TODO: 太慢了，看看怎么加速 (考虑多线程) eniops.rearrange
    start = desList[0]
    for des in desList[1:]:
        start = np.vstack((start,des))
    return start

def kMeans(data, n_training_samples=1024,n_clusters=200):
    # data = data.reshape(-1,128) # SIFT=128, ORB=32
    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=n_training_samples,verbose=1)
    model.fit(data)
    closest, _ = pairwise_distances_argmin_min(model.cluster_centers_.tolist(), data)
    vocabulary = data[closest] # 把聚类中心换成我们的数据
    # return model, model.cluster_centers_
    return model, vocabulary

# def KMeans_clusters_selctor(data):
#     model = MiniBatchKMeans(batch_size=2048 ,verbose=1)
#     # k is range of number of clusters.
#     visualizer = KElbowVisualizer(model, k=(490,510), timings= True) 
#     # visualizer = KElbowVisualizer(model, k=(2,30),metric='silhouette', timings= True)
#     visualizer.fit(data)        # Fit the data to the visualizer
#     visualizer.show() 

# [1500张图片， 200个聚类中心（词汇条目）]

def img2Hist(vocabulary, desList, image_counter, no_clusters,):
    img_hists = np.array([np.zeros(no_clusters) for _ in range(image_counter)])
    if image_counter == 1:
        feature = np.array(desList)
        # vq
        predict_idies, distance = vq.vq(feature, vocabulary)
        for idx in predict_idies:
            img_hists[0][idx] += 1 # （1/len(predict_idies)）

        return img_hists
    else:
        for i in trange(image_counter, desc='extracting feature per image'):
            feature = np.array(desList[i])
            # print(feature.shape)
            # feature = feature.reshape(-1, 128) # SIFT
            # feature = feature.reshape(-1, 32) # orb
            # vq
            predict_idies, distance = vq.vq(feature, vocabulary)
            for idx in predict_idies:
                img_hists[i][idx] += 1 # （1/len(predict_idies)）
    # vq 5个聚类中心，计算最近的数个，得到聚类中心set的下标（predict_idies）,
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

def svmClf(data, label):

    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.8, shuffle=True)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto')) 
    clf.fit(X_train, y_train)

    return clf, metrics.classification_report(y_test,clf.predict(X_test), target_names=labels)

def test(Path, clf, visual_words, n_clusters):
    results = []
    fileNames = os.listdir(Path)
    fileNames.sort(key= lambda x:int(x[:-4]))
    for imgPath in tqdm(fileNames,desc='Exporting test results...'):
        if(imgPath.startswith('.')): continue # Ignore the .DS_Stroe
        imgFullPath = os.path.join(Path, imgPath)
        sift = cv2.xfeatures2d.SIFT_create() # (xxx, 128)
        # orb = cv2.ORB_create() # (xxx, 32)
        kp, des = sift.detectAndCompute(readImg(imgFullPath),None)
        # reshape the matrix to vector
        imgFeature_test_CLF = img2Hist(visual_words, des, 1, n_clusters,)
        y_predicted = clf.predict(imgFeature_test_CLF)
        results.append(imgPath + ' ' + str(list(labels.keys())[list(labels.values()).index(y_predicted[0])]))
    
    f=open("results_run_3.txt","w")
    
    f.writelines('\n'.join(results))
    f.close()
    print('Done')

np.random.seed(42) 

n_clusters = 500
n_training_samples = 1024 # batch_size

imgVector_train, labelVector_train, img_counter = img2Kp_Des(trainingDatasetPath, )
imgVector_train_kmeans = list2vstack(list(imgVector_train.values()))

# print(imgVector_train.shape) # (630380, 128)
print('Training KMeans...')
kmeans, visual_words = kMeans(imgVector_train_kmeans, n_training_samples=n_training_samples, n_clusters=n_clusters)

# print(visual_words.shape)

print('Extracting features...')
# _, imgFeature_train = idf_and_norm(img2Hist(kmeans, imgVector_train, img_counter, n_clusters))
imgFeature_train = img2Hist(visual_words, imgVector_train, img_counter, n_clusters)
# _, imgFeature_train = idf_and_norm(imgFeature_train)
print('Training OvRLCs...')

final_model, score = svmClf(imgFeature_train, labelVector_train,)
print(score)
test(testDatasetPath, final_model,visual_words,n_clusters)
