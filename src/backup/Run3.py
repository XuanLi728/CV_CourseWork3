# coding: utf-8
import logging
import os

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

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

CONFIG = {
    "seed": 3047,
    "epochs":80,
    "img_size": 256,
    #           "model_name": "efficientnet_b3a",
    "model_1":"efficientnet_b3a",
    "model_2":"dla60_res2net",
    "model_3":"mobilenetv3_small_050",
    "model_4":"gluon_resnet34_v1b",
    "num_classes": 15,
    "train_batch_size": 32,
    "valid_batch_size": 32,
    "learning_rate": 1e-3,
    'T_0': 5,
    "eta_min": 1e-4,
    "T_max": 500,
    'T_mult':2,
    "weight_decay": 1e-6,
    "n_fold": 5,
    "n_accumulate": 1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "test_mode":True, # enable for testing pipeline, changes epochs to 2 and uses just 100 training samples
    "enable_amp_half_precision": False, # Try it in your local machine (the code is made for working with !pip install apex, not the pytorch native apex)
    # ArcFace Hyperparameters
    "s": 30.0, 
    "m": 0.30,
    "ls_eps": 0.0,
    "easy_margin": False
}

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

class SceneDatasets(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs 
            self.transform = transform
            self.target_transform = target_transform
    def __getitem__(self, index):
        imgPath, label = self.imgs[index]
        img = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            img = self.transform(img) 
        return img, label
    def __len__(self):
        return len(self.imgs)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = torch.cos(m)
        self.sin_m = torch.sin(m)
        self.th = torch.cos(torch.pi - m)
        self.mm = torch.sin(torch.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if(CONFIG['enable_amp_half_precision']==True):
            cosine = cosine.to(torch.float32)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=CONFIG['device'])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class SceneModel(nn.Module):
    def __init__(self, model_name, n_class, pretrained=True):
        super(SceneModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained,)
#         in_features = self.model.classifier.in_features
        in_features = self.model.fc.in_features
        self.model.reset_classifier(0)
#         self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.drop = nn.Dropout(p=0.2, inplace=False)
        self.fc = nn.Linear(in_features, n_class)

#         self.npc = NPCFace(512, CONFIG["num_classes"],)
    def forward(self, images, labels):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        pooled_drop = self.drop(pooled_features)
        emb = self.fc(pooled_drop)
        output = self.arc(emb,labels)
        return output,emb
    

trainingDatasetPath = 'data/training'
testDatasetPath = 'data/testing'

training_transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.Resize(255,255),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(0.45526364,0.24906044),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.45526364,0.24906044),
])

sceneDatasets = SceneDatasets(trainingDatasetPath,training_transform)

train_size = int(len(sceneDatasets) * 0.8) # 8:2 数据集分割
test_size = len(sceneDatasets) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(sceneDatasets, [train_size, test_size])

train_loader = DataLoader(train_dataset,batch_size=CONFIG['train_batch_size'], num_workers=4)
val_loader = DataLoader(val_dataset,batch_size=CONFIG['valid_batch_size'], num_workers=4)

model = SceneModel(CONFIG['model_4'])
model.to(CONFIG['device'])
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], )
# optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], 
#                        momentum=0.9)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], T_mult=CONFIG['T_mult'], eta_min=CONFIG['eta_min'])

logger = logging.getLogger("template_model.train")
logger.info("Start training")

for epoch in range(CONFIG['epochs']):
    # train
    for batch in train_loader:
        batch.to(CONFIG['device'])
        optimizer.zero_grad()
        optimizer.step()
        predicts = model(batch[0])
        loss = loss_fn(predicts,batch[1])
        # logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
        #                 .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss']))
        loss.backward()
        scheduler.step()

    if (epoch % 10)==0:
        with torch.no_grad():
            for batch in val_loader:
                batch.to(CONFIG['device'])
                predicts = model(batch[0])
                accuracy_score = metrics.accuracy_score(batch[1], predicts)
                report = metrics.classification_report(batch[1], predicts, target_names=labels)
