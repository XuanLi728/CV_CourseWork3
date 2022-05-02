
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
    training_accuracy = metrics.accuracy_score(clf.predict(X_train),y_train)
    val_accuracy = metrics.accuracy_score(clf.predict(X_test),y_test)
    # training_acc = metrics.classification_report(clf.predict(X_train), y_train, target_names=labels)
    # val_acc = metrics.classification_report(clf.predict(X_test), y_test, target_names=labels)
    return clf, training_accuracy, val_accuracy

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

    def plot_figure(ax, acc_list, acc_type, model_list):
        top_index = np.argmax(acc_list)
        top_acc = np.max(acc_list)

        min_index = np.argmin(acc_list)
        min_acc = np.min(acc_list)
        s = 'Top ' + acc_type + ' Acc: (' + str(model_list[top_index]) + ',' + str(round(top_acc, 2)) + ')'
        s2 = 'Min ' + acc_type + ' Acc: (' + str(model_list[min_index]) + ',' + str(round(min_acc,2)) + ')'

        ax.set_xlabel('number of models ' + acc_type)
        ax.set_ylabel('accuracy')
        ax.grid()

        ax.plot(model_list, acc_list)
        # ax.set_xticks(ticks=model_list,)
        ax.plot(model_list[top_index],top_acc, 'rX')
        # X_location = 25 if model_list[top_index] == 30 else model_list[top_index]
        ax.text(model_list[top_index],top_acc, s)

        ax.plot(model_list[min_index],min_acc, 'gX')
        # X_location = 25 if model_list[min_index] == 30 else model_list[min_index]
        ax.text(model_list[min_index],min_acc, s2)

    train_acc = []
    val_acc = []
    n_neighbors_list = np.arange(2,50)
    best_acc = 0
    for n_model in tqdm.tqdm(n_neighbors_list, desc='training with different n_neighbors'):
        clf, train_accuracy, val_accuracy = train(imgFeature_train, labelVector_train, n_model)
        if val_accuracy >= best_acc:
            best_model = clf 
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)

    import matplotlib.pyplot as plt
    train_acc_figure = plt.subplot(2, 1, 1, frameon = False) # 两行一列，位置是1的子图
    val_acc_figure = plt.subplot(2, 1, 2, frameon = False) 
    plot_figure(train_acc_figure, train_acc, 'train', n_neighbors_list)
    plot_figure(val_acc_figure, val_acc, 'val', n_neighbors_list)
    plt.tight_layout()
    plt.show()
    return best_model

def main():
    print('training...')
    # load the data
    imgFeature_train, labelVector_train = img2matrix(trainingDatasetPath)
    np.random.seed(7)
    # n_neighbors=19 # 27 Acc
    # clf, training_acc, val_acc = train(imgFeature_train, labelVector_train, n_neighbors)
    # print(training_acc)
    # print(val_acc)
    best_model = tuning(imgFeature_train, labelVector_train)
    print('testing...')
    test(testDatasetPath, best_model)


if __name__ == "__main__":
    main()

