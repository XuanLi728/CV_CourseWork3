import os

import albumentations as A
import cv2
import numpy as np
import sklearn
from PIL import ImageEnhance
from scipy.cluster import vq
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm, trange
from yellowbrick.cluster import KElbowVisualizer

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

np.random.seed(42) 

img_size=256
n_clusters = 500
n_training_samples = 1024 # batch_size
Aug_times = 5 # 1->48; 2->69; 3->81; 4->0.88

'''
    Image Augmentation
'''

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# not using 
# test time augmentation
tta_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(p=0.5),
    A.RandomRotate90(p=0.2),
])

# 读图片（readImg） ——> img2Kp_Des(特征提取) -> kmeans (得到词汇本 vocabulary) -> 把图像特征转换成histograms(img2Hist) -> 训练(svnClf)

# #直方图均衡化
# enhanced_gray = cv2.equalizeHist(img) if proabilityControl(0.5) else img 
# imgAugSet.append(enhanced_gray)


'''
Data Augmentation and Data Multiplication. 
Each time, apply data augmentation to the same image, 
append the processed image in imgList as a new image.

input:
    img: image need to be store (ndarray)
    times:int, Data amplification times (int)
output:
    imgList: list, list of image(ndarray)
'''

def dataAug(img, times):
    imgList = []
    for _ in range(times):
        # imgList.append(transform(image=img)['image'])
        imgList.append(img)
    return imgList

'''
use MinMaxScaler normalize histogram
input:
    hist: Histogram before normalize (ndarray)
output:
    hist: histogram after normalize (ndarray)
'''

def histNorm(hist):
    scaler = MinMaxScaler().fit(hist)
    hist = scaler.transform(hist)
    return hist

'''
Use StandardScaler normalization data
input:
    x: data needs to be normalize (ndarray)
output:
    image: data that has been normalized (ndarray)
'''
def normalisation(x):  
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    image = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return image
    
'''
get image data
input: 
    path:the path of image (string)
output:
    img: images (ndarray)
'''
def readImg(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = normalisation(img) # 归一化 zero mean and unit variance
    img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_NEAREST)
    return img

'''
Feature extraction
input:
    path:path of dataset (string)
    times: Data amplification times (int)
output:
    imgVector: vector shape data of image (ndarray)
    labelvector: label of image (ndarray)
    img_counter: total numer of images (int)
'''
def img2Kp_Des(Path,times):
    train_desDict = {}
    val_desDict = {}
    labelVector_train = []
    labelVector_val = []

    # class_counter = np.zeros((15, ), dtype=int)
    img_counter_train = 0 
    img_counter_val = 0
    for dirName in tqdm(os.listdir(Path), desc='reading images...'):
        if(dirName.startswith('.')): continue # Ignore the .DS_Stroe
        dirFullPath = os.path.join(Path, dirName)

        class_index = labels[dirName]
        class_counter = 0
        for imgPath in os.listdir(dirFullPath):
            class_total_image = len(os.listdir(dirFullPath))
            train_sample = int(class_total_image * 0.8) # for dataset expanding w/o ImgAug
            if(imgPath.startswith('.')): continue # Ignore the .DS_Stroe

            imgFullPath = os.path.join(dirFullPath, imgPath)
            img_mat = readImg(imgFullPath) # read image

            class_counter += 1 
            if class_counter >= train_sample: # testing part of image for training (w/o Image Aug)
                extractor = cv2.SIFT_create() # (xxx, 128)
                # extractor = cv2.ORB_create() # (xxx, 32)
                kp, des = extractor.detectAndCompute(img_mat,None)
                val_desDict[img_counter_val] = des
                labelVector_val.append(class_index)
                img_counter_val += 1
            else: # for training (w/o Image Aug)
                for augImg in dataAug(img_mat, times=times):
                    extractor = cv2.SIFT_create() # (xxx, 128)
                    # extractor = cv2.ORB_create() # (xxx, 32)
                    kp, des = extractor.detectAndCompute(augImg,None)

                    train_desDict[img_counter_train] = des
                    labelVector_train.append(class_index)
                    img_counter_train += 1

    return (
        train_desDict, 
        list2vstack(labelVector_train), 
        val_desDict, 
        list2vstack(labelVector_val),
        img_counter_train, 
        img_counter_val
    )
'''
two way to change list[ndarray] type data to ndarray 
list2vstack: using vstack
list2vstack1: using normal way with change type and reshape
input:
    desList: data that should be change (list[ndarray])
output:
    start: data that change to ndarray type (ndarray ([[n1,128],
                                                        [n1,128],
                                                        ...
                                                        [m1, 128]]))
'''
def list2vstack(desList):
    start = desList[0]
    for des in tqdm(desList[1:],desc='list stacking'):
        start = np.vstack((start,des))
    return start

def list2vstack1(desList):
    start = desList[0].tolist()
    for des in desList:
        for i in des:
            start.append(i)
    start = np.array(start)
    start.reshape(-1,128)
    return start

'''
Use miniBatch way to run KMeans 
input:
    data: stacking SIFT features (ndarray)
    n_training_sample: the number of training sample, i.e. batch size (int)
    n_clusters: the number of class (int)
output:
    model: trained K means model (sklearn clf object: Fitted estimator.)
    vocabulary: Vocabularies correspond to categories of SIFT features, i.e. cluster centres (ndarray [n_clusters, 128])
'''
def kMeans(data, n_training_samples=1024,n_clusters=200):
    # data = data.reshape(-1,128) # SIFT=128, ORB=32
    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=n_training_samples,verbose=1)
    model.fit(data)
    closest, _ = pairwise_distances_argmin_min(model.cluster_centers_.tolist(), data)
    vocabulary = data[closest] # 把聚类中心换成我们的数据
    # return model, model.cluster_centers_
    return model, vocabulary

def KMeans_clusters_selctor(data):
    model = MiniBatchKMeans(batch_size=1024 ,verbose=1)
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=[15, 400, 500, 600], timings= True) 
    # visualizer = KElbowVisualizer(model, k=[15, 400, 500, 600],metric='silhouette', timings= True) cannot work
    visualizer.fit(data)        # Fit the data to the visualizer
    visualizer.show() 


'''
[1500 images, 200(hyperparamter) clustering centers (vocabulary items)]

Use vq.vq to get histogram of image
Vq 5 cluster centers, calculate the nearest number, get the subscript of cluster center set(predict_idies)
input:
    vocabulary: Vocabularies correspond to categories of image slices (ndarray [no_clusters, 128])
    desList: ndarray of SIFT features (ndarray) 
    image_counter: number of image (int)
    no_clusters: number of class (int)
output:
    img_hists: image histogram (ndarray, [image_counter, no_clusters])
'''
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
        return img_hists


'''
The occurrence frequency of the word in the dataset is regarded as a weight, 
and the word frequency vector is processed once
The word frequency vector is normalized and then compared
input:
    img_historgram_list: the histograms of image (ndarray, [image_counter, no_clusters])
output:
    idf:Word frequency vector (not using in the rest code)
    img_book_list: codebook(histograms) for training (ndarray, [image_counter, no_clusters])
'''
def idf_and_norm(img_histogram_list):
    words_occurrences = np.sum((img_histogram_list > 0) * 1, axis=0)  # 求出每个word在所有图片中出现的频数
    idf = np.array(np.log((1.0 * len(img_histogram_list) + 1) / (1.0 * words_occurrences + 
         1)),'float32')     
 
    img_book_list = img_histogram_list * idf
    img_book_list = sklearn.preprocessing.normalize(img_book_list, norm='l2')  # 归一化
    return idf, img_book_list

'''
SVM hyperparamters tunning functions
'''

def svcParamSelection(X, y,):
    # best : 0.1, 0.5

    param_grid = {
            'kernel':['linear','rbf','sigmoid','poly'],
            'C':np.linspace(1e-8,1,9),
            'gamma':np.linspace(0.1,10,5)
        }

# {'C': 0.2500000075, 'gamma': 0.1, 'kernel': 'linear'}
    # clf = make_pipeline(StandardScaler(), SVC()) 
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=4, verbose=1,return_train_score=True)

    grid_search.fit(X, y)

    import pandas as pd 
    cv_result = pd.DataFrame.from_dict(grid_search.cv_results_) 
    with open('cv_result.csv','w') as f: 
        cv_result.to_csv(f)

    print(grid_search.best_params_)


'''
using train_test_split to randomly divide training set and test set 
to train and val the model 
input:
    data: input data to be divide to training set and test set (ndarray)
    label: label of each picture in data (ndarray)
output:
    clf: Trained model (sklearn clf object: Fitted estimator.)
    report: Classification report of training process (string)
'''
def svmClf(data, label):

    # X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=1, shuffle=True)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    clf = make_pipeline(StandardScaler(), SVC(C=0.25, gamma= 0.1, kernel='linear')) 
    clf.fit(data, label)

    training_acc = metrics.classification_report(clf.predict(data), label, target_names=labels)
    # val_acc = metrics.classification_report(clf.predict(X_test), y_test, target_names=labels)
    return clf, training_acc

'''
using train_test_split to randomly divide training set and test set 
and train model 
input:
    data: ndarray, input data to be divide to training set and test set
    label: label of each picture in data
    n_models: number of models(only use in training mutiple model)
output:
    clf: Trained model (sklearn clf object: Fitted estimator.)
    report:Classification report of training process (string)
'''
def OvRLCs(data, label, n_models):

    X_train, y_train = train_test_split(data, label, train_size=0.8, shuffle=True)
    estimators = []
    for index in range(n_models):
        # model = ('svc_'+str(index), SVC(C=0.5, gamma=0.1))
        # model = ('lr_'+str(index), LinearSVC(multi_class='ovr'))
        model = ('lr_'+str(index), LogisticRegression())
        estimators.append(model)

    clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=5,
        verbose=1,
        n_jobs=4,
    )
    clf.fit(X_train, y_train)

    train_report = metrics.classification_report(y_train,clf.predict(X_train), target_names=labels)
    return clf, train_report

def val(data, label, clf ,img_counter_val, visual_words,n_clusters):
    imgFeature_val = img2Hist(visual_words, data, img_counter_val, n_clusters)

    # X_val, _, y_val, _ = train_test_split(imgFeature_val, label, train_size=1,shuffle=True)
    train_report = metrics.classification_report(label,clf.predict(imgFeature_val), target_names=labels)
    
    return train_report

'''
test model and get a txt test result
input:
    Path: path of dataset
    clf: Trained model
    visual_words:Visual word vector dictionary
    n_clusters: number of clusters
   
'''
def test(Path, clf, visual_words, n_clusters):
    results = []
    fileNames = os.listdir(Path)
    fileNames.sort(key= lambda x:int(x[:-4]))
    for imgPath in tqdm(fileNames,desc='Exporting test results...'):
        if(imgPath.startswith('.')): continue # Ignore the .DS_Stroe
        imgFullPath = os.path.join(Path, imgPath)
        sift = cv2.SIFT_create() # (xxx, 128)
        # orb = cv2.ORB_create() # (xxx, 32)
        kp, des = sift.detectAndCompute(readImg(imgFullPath),None)
        # reshape the matrix to vector
        imgFeature_test_CLF = img2Hist(visual_words, des, 1, n_clusters,)
        y_predicted = clf.predict(imgFeature_test_CLF)
        results.append(
            imgPath + ' ' + str(
                list(labels.keys())[
                        list(labels.values()).index(y_predicted[0])
                    ]
                ).lower()
            )

    f=open("results_run_3.txt","w")
    
    f.writelines('\n'.join(results))
    f.close()
    print('Done')

def main():
    imgVector_train, labelVector_train, imgVector_val, labelVector_val, img_counter_train, img_counter_val = img2Kp_Des(trainingDatasetPath, Aug_times)
    imgVector_train_kmeans = list2vstack1(list(imgVector_train.values()))
    # print(imgVector_train.shape) # (630380, 128)
    print('Training KMeans...')
    kmeans, visual_words = kMeans(imgVector_train_kmeans, n_training_samples=n_training_samples, n_clusters=n_clusters)
    # KMeans_clusters_selctor(imgVector_train_kmeans)
    # print(visual_words.shape)

    print('Extracting features...')
    imgFeature_train = img2Hist(visual_words, imgVector_train, img_counter_train, n_clusters)
    # _, imgFeature_train = idf_and_norm(imgFeature_train)
    print('Training SVM...')
    # svcParamSelection(imgFeature_train, labelVector_train,)
    final_model, train_score = svmClf(imgFeature_train, labelVector_train,) # 81
    print(train_score)
    val_score = val(
        data=imgVector_val, 
        label=labelVector_val,
        clf=final_model,
        img_counter_val=img_counter_val,
        visual_words=visual_words,
        n_clusters=n_clusters)

    print(val_score)
    # final_model, score = OvRLCs(imgFeature_train, labelVector_train,15) # 41
    # test(testDatasetPath, final_model,visual_words,n_clusters)

if __name__ == "__main__":
    main()
