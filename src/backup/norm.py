import os

import cv2
import numpy as np

# img_h, img_w = 32, 32
img_h, img_w = 255, 255   #根据自己数据集适当调整，影响不大
means, stdevs = [], []
img_list = []
count_dict = {}
imgs_path = 'data/training'
imgs_path_list = os.listdir(imgs_path)

len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    if(item.startswith('.')): continue 
    img_folder = os.path.join(imgs_path,item)
    for imgPath in os.listdir(img_folder):
        if(imgPath.startswith('.')): continue 
        img = cv2.imread(os.path.join(img_folder,imgPath))
        # print(os.path.join(img_folder,imgPath))
        for x in img.shape:
            if x in count_dict:
                count_dict[x] += 1
            else:
                count_dict[x] = 1
        img = cv2.resize(img,(img_w,img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        print(i,'/',len_)    
 
imgs = np.concatenate(img_list, axis=2)
imgs = imgs.astype(np.float32) / 255.
 
# for i in range(3):
pixels = imgs[:, :].ravel()  # 拉成一行
means.append(np.mean(pixels))
stdevs.append(np.std(pixels))
 
# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()
 
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print(count_dict)
import matplotlib.pylab as plt

lists = sorted(count_dict.items()) # sorted by key, return a list of tuples
# normMean = [0.45526364]
# normStd = [0.24906044]
# 255x255最多
# x, y = zip(*lists) # unpack a list of pairs into two tuples

# plt.plot(x, y)
# plt.show()
# print(lists)
