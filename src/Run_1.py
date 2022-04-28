
import os
import tqdm
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

'''
get image data
input: 
    path:string, the path store images
output:
    img:ndarray, images 
'''
def readImg(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    length = min(img.shape[0], img.shape[1])

    x = img.shape[1] // 2 - length//2
    y = img.shape[0] // 2 - length//2

    img = img[y:y+length, x:x+length]
    img = cv2.resize(img,(16,16),interpolation=cv2.INTER_NEAREST)
    return img


'''
get image feature and lavel
read all image and get this images' feature and images' Corresponding label
input: 
    path:string, the path store images
output:
    imgFeature: ndarray, feature of image
    labelVector: ndarray, label of each image
'''
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


'''
training model
split the datasets into training and testing parts，then feed to  a knn classifier 
input: 
    imgFeature_train: ndarray, image feature of training set
    labelVector_train: ndarray, label of training set
    n_neighbors: Kmeans parameter,Number of neighbors to use by default for kneighbors queries.
output:
    clf: Classifier implementing the k-nearest neighbors vote.
    accuracy:score of Kmeans classifier
'''
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


'''
Build text reports that show the main classification indicators
input:
    groudTruth:1d array-like, or label indicator array / sparse matrix.Ground truth (correct) target values.
    predicted:1d array-like, or label indicator array / sparse matrix. Estimated targets as returned by a classifier.
output:
    reportstr or dict.Text summary of the precision, recall, F1 score for each class. Dictionary returned if output_dict is True. 
'''
def Val(groudTruth, predicted):
    return metrics.classification_report(groudTruth, predicted, target_names=labels)


'''
Test the model using test sets and enter test results.
input:
    Path:string, path of test address
    clf: The trained Kmeans model
'''
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

def main():
    # load the data
    imgFeature_train, labelVector_train = img2matrix(trainingDatasetPath)
    # x_train, x_val, y_train, y_val = train_test_split(imgFeature, labelVector, train_size=0.8, random_state=42)
    # clf, scores = train(x_train, x_val, y_train, y_val)
    # print(scores)
    np.random.seed(42)
    # n_neighbors=5 # 25 Acc
    # clf, accuracy = train(imgFeature_train, labelVector_train, n_neighbors)
    # test(testDatasetPath, clf)
    # print(len(os.listdir(testDatasetPath))) # A total of 2985 samples were tested
    acc = []
    for n_neighbours in tqdm.tqdm(np.arange(start=1,stop=100)):
        clf, accuracy = train(imgFeature_train, labelVector_train, n_neighbours)
        acc.append(accuracy)
    import matplotlib.pyplot as plt

    plt.plot(np.arange(1,100), acc)
    plt.show()


if __name__ == "__main__":
    main()

