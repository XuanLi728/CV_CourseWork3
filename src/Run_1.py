
import os

import cv2
import numpy as np
import tqdm
from sklearn import metrics
from sklearn.model_selection import train_test_split
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
    # centre crop
    x = img.shape[1] // 2 - length//2
    y = img.shape[0] // 2 - length//2

    img = img[y:y+length, x:x+length]
    img = cv2.resize(img,(16,16),interpolation=cv2.INTER_NEAREST)
    return img


'''
get image features and labels
read all image and get this images' features and images' Corresponding labels
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
split the datasets into training and testing parts, then feed to  a knn classifier 
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

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # training_accuracy = metrics.accuracy_score(y_predicted,y_test)
    training_acc = metrics.classification_report(clf.predict(X_train), y_train, target_names=labels)
    val_acc = metrics.classification_report(clf.predict(X_test), y_test, target_names=labels)
    return clf, training_acc, val_acc

'''
Test the model using test sets and return test results.
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
        # transform the predicted label(int) into the corresponding label(str), return in lower letters
        results.append(
            imgPath + ' ' + str(
                list(labels.keys())[
                        list(labels.values()).index(y_predicted[0])
                    ]
                ).lower()
            )
    
    f=open("results_run_1.txt","w")
    
    f.writelines('\n'.join(results))
    f.close()
    print('Done')

def tuning(imgFeature_train, labelVector_train):
    acc = []
    for n_neighbours in tqdm.tqdm(np.arange(start=1,stop=100)):
        clf, accuracy = train(imgFeature_train, labelVector_train, n_neighbours)
        acc.append(accuracy)
    import matplotlib.pyplot as plt

    top_index = np.argmax(acc)
    top_acc = np.max(acc)

    min_index = np.argmin(acc)
    min_acc = np.min(acc)
    s = 'Top Acc: (' + str(top_index) + ',' + str(top_acc) + ')'
    s2 = 'Min Acc: (' + str(min_index) + ',' + str(min_acc) + ')'

    plt.plot(np.arange(1,100), acc)
    plt.plot(top_index,top_acc, 'rX')
    plt.text(top_index,top_acc, s)

    plt.plot(min_index,min_acc, 'gX')
    plt.text(min_index,min_acc, s2)
    plt.show()

def main():
    print('training...')
    # load the data
    imgFeature_train, labelVector_train = img2matrix(trainingDatasetPath)
    np.random.seed(7)
    n_neighbors=5 # 25 Acc
    clf, training_acc, val_acc = train(imgFeature_train, labelVector_train, n_neighbors)

    print(training_acc)
    print(val_acc)

    print('testing...')
    test(testDatasetPath, clf)


if __name__ == "__main__":
    main()

