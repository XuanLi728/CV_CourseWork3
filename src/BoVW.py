import os

import cv2
import numpy as np
import sklearn
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
    return scaler.transform(x)

def readImg(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    # img = normalisation(img) # 归一化 zero mean and unit variance
    img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_NEAREST)
    return img

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0]-windowSize[0], stepSize):
		for x in range(0, image.shape[1]-windowSize[1], stepSize):
			# yield the current window
			yield (image[y:y + windowSize[1], x:x + windowSize[0]])

def split_patchs(img, patch_size=8, down_sample_rate=2, step_size=2):
    patches = []
    if img.shape != (img_size,img_size):
        img = cv2.resize(img,(img_size,img_size))

    # for block in np.reshape(view_as_blocks(img, block_shape=(1,patch_size, patch_size)), (-1, patch_size, patch_size)): # 将图片分割为8x8的窗口
    #     print(block)
    #     patches.append(np.reshape(block[::4,::4], (-1,))) # 每隔4个采样，拉直 （无重叠）
    for epoch in range(down_sample_rate+1): # 金字塔降采样
        if epoch == 0:
            continue
        else:
            img = cv2.pyrDown(img)
        for patch in sliding_window(img, step_size, (patch_size,patch_size)):
            patches.append(np.reshape(patch[::4,::4], (-1,))) # 每隔4个采样，拉直 (有重叠)
    return np.array(patches)

def img2vectors(Path, step_size):
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
            img_vectors = normalisation(split_patchs(readImg(imgFullPath), step_size)) # 每张图片分割为 n x (4, )的vector
            img_counter += 1
            imgVector.append(img_vectors)
            labelVector.append(class_index)
        class_counter[class_index] += img_counter
    return np.array(imgVector), np.array(labelVector), np.sum(class_counter, dtype=int), class_counter

def kMeans(data, n_training_samples=1024,n_clusters=492):

    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=n_training_samples,verbose=1)
    # for _ in trange(epochs):
    #     training_data = sample_without_replacement(data.shape[0], n_training_samples)
    model.fit(data)
    return model, model.cluster_centers_
# https://blog.csdn.net/qq_36622009/article/details/102895411
def idf_and_norm(img_histogram_list):
    words_occurrences = np.sum((img_histogram_list > 0) * 1, axis=0)  # 求出每个word在所有图片中出现的频数
    idf = np.array(np.log((1.0 * len(img_histogram_list) + 1) / (1.0 * words_occurrences + 
         1)),'float32')     
 
    img_book_list = img_histogram_list * idf
    img_book_list = sklearn.preprocessing.normalize(img_book_list, norm='l2')  # 归一化
    return idf, img_book_list

# def KMeans_clusters_selctor(data):
#     model = MiniBatchKMeans(batch_size=2048 ,verbose=1)
#     # k is range of number of clusters.
#     visualizer = KElbowVisualizer(model, k=(490,510), timings= True) 
#     # visualizer = KElbowVisualizer(model, k=(2,30),metric='silhouette', timings= True)
#     visualizer.fit(data)        # Fit the data to the visualizer
#     visualizer.show() 
    
def img2Feature(model, Path, image_counter, no_clusters,):
    im_features = np.array([np.zeros(no_clusters) for _ in range(image_counter)])
    imgVector = []
    labelVector = []
    for i in trange(image_counter, desc='extracting feature per image'):
        feature = imgVector_train[i]
        feature = feature.reshape(-1, 4)
        idx = model.predict(feature)
        im_features[i][idx] += 1
    # # TODO: 将之前读取完的特征重用
    # for dirName in tqdm(os.listdir(Path), desc='extracting feature per class...'):
    #     if(dirName.startswith('.')): continue # Ignore the .DS_Stroe
    #     dirFullPath = os.path.join(Path, dirName)
    #     img_counter = 0
    #     class_index = labels[dirName]

    #     for imgPath in os.listdir(dirFullPath):
    #         if(imgPath.startswith('.')): continue # Ignore the .DS_Stroe

    #         imgFullPath = os.path.join(dirFullPath, imgPath)
    #         img_vectors = normalisation(split_patchs(readImg(imgFullPath))) # 每张图片分割为 n x (4, )的vector
    #         feature = img_vectors.reshape(-1, 4)
    #         idx = model.predict(feature)
    #         im_features[img_counter][idx] += 1
    #         labelVector.append(class_index)
    #         img_counter+=1

    return im_features

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

img_size = 256 #越小获取到的信息可能更多
n_clusters = 600
n_models = 15
n_training_samples = 1024 # batch_size
step_size = 3

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

imgVector_train, labelVector_train, img_counter, class_counter = img2vectors(trainingDatasetPath, step_size=step_size)
imgVector_train = np.reshape(imgVector_train, (-1,4))
# print(imgFeature_train.shape) # (1536000, 4)
print('Training KMeans...')
kmeans, visual_words = kMeans(imgVector_train, n_training_samples=n_training_samples, n_clusters=n_clusters)
# KMeans_clusters_selctor(imgVector_train)
print('Extracting features...')
imgFeature_train = img2Feature(kmeans, trainingDatasetPath, img_counter, n_clusters)
# idf, imgFeature_train = idf_and_norm(imgFeature_train)
# imgVector_train = normalisation(imgFeature_train)
# print(imgFeature_train.shape) #(1500, 500)
print('Training OvRLCs...')
final_model, score = OvRLCs(imgFeature_train, labelVector_train, n_models=n_models)
print(score)
