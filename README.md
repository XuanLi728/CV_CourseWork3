## 配置
  1. 文件树配置
  2. 环境配置
     1. conda env create -f environment.yaml
     2. pip install -r requirnments.txt

## TODO：
- [X] Run 1 KNN
  - 当前 Acc: 25% n=5
  - 交叉验证Acc 25%, cv=5, train_size=0.9
  - [ ] 注释
  - [ ] 添加preprocess
  - [ ] 代码结构美化
  - [ ] 添加PCA，KNN调参

- [ ] Run 2 Bag-of-visual-words
  - 当前 Acc: 0.06 (???)
  - [ ] Debug, 找出表现差的原因
  - [X] 添加preprocess
  - [X] 添加滑动窗口patches提取
  - [X] 添加Img2Vector
  - [X] 添加Img2Feature
  - [x] 添加线形分类器集合

- [ ] Run 3 SIFT (冲点)
  - [ ] 使用先前的场景识别模型微调
  - [ ]


## 资源
https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj4/html/euzun3/index.html

http://openimaj.org/tutorial/classification101.html

https://github.com/gary1346aa/Scene-Recognition

https://github.com/gurkandemir/Bag-of-Visual-Words

https://github.com/AMANVerma28/Indoor-Outdoor-scene-classification

https://github.com/buptchan/scene-classification

- Run3
  - https://blog.csdn.net/qq_36622009/article/details/102895411
  - https://blog.csdn.net/cliukai/article/details/102525486
  - https://blog.csdn.net/weixin_45666660/article/details/109104700
  - （代码示例）https://www.kaggle.com/code/pierre54/bag-of-words-model-with-sift-descriptors/notebook
  - (优化相关)https://github.com/shackenberg/phow_caltech101.py 
  - （优化相关, ORB算法）https://zhuanlan.zhihu.com/p/261966288