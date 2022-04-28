import os
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

#Adjustable global parameters
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
img_size = 256 #The smaller you are, the more information you get
n_clusters = 500
n_models = 15
n_training_samples = 1024 # batch_size
step_size = 3

'''
Use MinMaxScaler normalization data
input:
    x:the data should be normalize
output:
    dta been normalized
'''
def normalisation(x):  
    scaler = MinMaxScaler().fit(x)
    return scaler.transform(x)

'''
get image data
input: 
    path:string, the path store images
output:
    img:ndarray, images 
'''
def readImg(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    # img = normalisation(img) # normalization zero mean and unit variance
    img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_NEAREST)
    return img

'''
Slide the window over the image
input:
    image: ndnarry, The image of the sliding window
    stepSize: int, The pixel distance of each window slide
    windowSize: int, Size of window
'''
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0]-windowSize[0], stepSize):
		for x in range(0, image.shape[1]-windowSize[1], stepSize):
			# yield the current window
			yield (image[y:y + windowSize[1], x:x + windowSize[0]])

'''
Image segmentation and downsampling
input:
    img:ndarray, image be segmented
    patch_size:int, Image segmentation size
    down_sample_rate: int, The interval between samples
    step_size: int, Step size of downsampling
output:
    patches: ndarray, Every 4 samples, straightened (no overlap)
'''
def split_patchs(img, patch_size=8, down_sample_rate=2, step_size=2):
    patches = []
    if img.shape != (img_size,img_size):
        img = cv2.resize(img,(img_size,img_size))

    for block in np.reshape(view_as_blocks(img, block_shape=(patch_size, patch_size)), (-1, patch_size, patch_size)): # Split the image into 8x8 Windows
        patches.append(np.reshape(block[::4,::4], (-1,))) # Every 4 samples, straightened (no overlap)
    # for epoch in range(down_sample_rate+1): # 金字塔降采样
    #     if epoch == 0:
    #         continue
    #     else:
    #         img = cv2.pyrDown(img)
    #     for patch in sliding_window(img, step_size, (patch_size,patch_size)):
    #         patches.append(np.reshape(patch[::4,::4], (-1,))) # 每隔4个采样，拉直 (有重叠)
    return np.array(patches)

'''
change image matrix to vextors 

input:
    path:string, image path
    step_size: int, the length of each step (ignore)
output:
    imgVector: ndarray, vector shape data of image
    labelvector: ndarray, label of image
    img_counter: the number of image be change to vector
'''
def img2vectors(Path, step_size):
    imgVector = {}
    labelVector = []

    img_counter = 0
    for dirName in tqdm(os.listdir(Path), desc='reading images...'):
        if(dirName.startswith('.')): continue # Ignore the .DS_Stroe
        dirFullPath = os.path.join(Path, dirName)
        class_index = labels[dirName]

        for imgPath in os.listdir(dirFullPath):
            if(imgPath.startswith('.')): continue # Ignore the .DS_Stroe
            imgFullPath = os.path.join(dirFullPath, imgPath)
            img_vectors = normalisation(split_patchs(readImg(imgFullPath), patch_size=8)) # 每张图片分割为 n x (4, )的vector
            imgVector[img_counter] = img_vectors
            labelVector.append(class_index)
            img_counter += 1
        # class_counter[class_index] += img_counter
    return imgVector, np.array(labelVector), img_counter, 

'''
two way to change list[ndarray] type data to ndarray 
list2vstack: using vstack
list2vstack1: using normal way with change type and reshape
input:
    desList: list[ndarray], data that should be change
output:
    start: ndarray, data that change to ndarray type
'''
def list2vstack(desList):
    start = desList[0]
    for des in desList[1:]:
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
    data: ndarray, data of calssification
    n_training_sample: int, the number of training sample
    n_clusters:int, the number of class
output:
    model:the K means model
    vocabulary: Vocabularies correspond to categories of image slices
'''
def kMeans(data, n_training_samples=1024,n_clusters=200):
    # data = data.reshape(-1,128) # SIFT=128, ORB=32
    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=n_training_samples,verbose=1)
    model.fit(data)
    closest, _ = pairwise_distances_argmin_min(model.cluster_centers_.tolist(), data)
    vocabulary = data[closest] # Change the clustering center to one of data

    return model, vocabulary


'''
[1500 images, 200 clustering centers (vocabulary items)]
Use vq.vq to get histogram of image
Vq 5 cluster centers, calculate the nearest number, get the subscript of cluster center set（predict_idies）
input:
    vocabulary: Vocabularies correspond to categories of image slices
    desList: list[ndarray], data 
    image_counter:int, number of image
    no_clusters: int, number of class
output:
    img_hists: image histogram
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
    # Vq 5 cluster centers, calculate the nearest number, get the subscript of cluster center set（predict_idies）,
        return img_hists


# https://blog.csdn.net/qq_36622009/article/details/102895411

'''
The occurrence frequency of the word in the dataset is regarded as a weight, 
and the word frequency vector is processed once
The word frequency vector is normalized and then compared
input:
    img_historgram_list: list, the histograms of image
output:
    idf:Word frequency vector
    img_book_list:The processed IMg_book_list is saved to the model
'''
def idf_and_norm(img_histogram_list):
    words_occurrences = np.sum((img_histogram_list > 0) * 1, axis=0)  # Figure out how often each word appears in all the pictures
    idf = np.array(np.log((1.0 * len(img_histogram_list) + 1) / (1.0 * words_occurrences + 
         1)),'float32')     
 
    img_book_list = img_histogram_list * idf
    img_book_list = sklearn.preprocessing.normalize(img_book_list, norm='l2')  # normalized
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

'''
by using train_test_split to randomly divide training set and test set 
and train model 
input:
    data: ndarray, input data to be divide to training set and test set
    label: label of each picture in data
    n_models: number of models(only use in training mutiple model)
output:
    clf: Trained model
    report:Classification report of training process
'''
def OvRLCs(data, label, n_models):

    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.9, shuffle=True)
    estimators = []
    for index in range(n_models):
        # model = ('svc_'+str(index), SVC(C=0.5, gamma=0.1))
        # model = ('lr_'+str(index), LinearSVC(multi_class='ovr'))
        model = ('lr_'+str(index), LogisticRegression())
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

        # reshape the matrix to vector
        imgFeature_test_CLF = img2Hist(visual_words, normalisation(split_patchs(readImg(imgFullPath), patch_size=8)), 1, n_clusters)
        y_predicted = clf.predict(imgFeature_test_CLF)
        results.append(imgPath + ' ' + str(list(labels.keys())[list(labels.values()).index(y_predicted[0])]).lower())
    
    f=open("results_run_2.txt","w")
    
    f.writelines('\n'.join(results))
    f.close()
    print('Done')
def main():


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

    imgVector_train, labelVector_train, img_counter, = img2vectors(trainingDatasetPath, step_size=step_size)
    imgVector_train_kmeans = list2vstack1(list(imgVector_train.values()))
    # print(imgFeature_train.shape) # (1536000, 4)
    print('Training KMeans...')
    kmeans, visual_words = kMeans(imgVector_train_kmeans, n_training_samples=n_training_samples, n_clusters=n_clusters)
    # KMeans_clusters_selctor(imgVector_train)
    print('Extracting features...')
    imgFeature_train_CLF = img2Hist(visual_words, imgVector_train, img_counter, n_clusters)
    _, imgFeature_train_CLF = idf_and_norm(imgFeature_train_CLF)
    # imgVector_train = normalisation(imgFeature_train)
    # print(imgFeature_train.shape) #(1500, 500)
    print('Training OvRLCs...')
    final_model, score = OvRLCs(imgFeature_train_CLF, labelVector_train, n_models=n_models) # 42
    print(score)
    print('Exporting test results...')
    test(testDatasetPath, final_model,visual_words,n_clusters)

if __name__ == "__main__":
    main()
