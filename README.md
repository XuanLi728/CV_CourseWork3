## TODO：
- [X] Run 1 KNN
  - 当前 Acc: 30.8%
  - 交叉验证Acc 最高只有17%, cv=5, train_size=0.8
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



## 资源
https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj4/html/euzun3/index.html

http://openimaj.org/tutorial/classification101.html

https://github.com/gary1346aa/Scene-Recognition

https://github.com/gurkandemir/Bag-of-Visual-Words