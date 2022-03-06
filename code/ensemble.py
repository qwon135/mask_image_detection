import random
import os, sys
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from datetime import datetime

import albumentations as A
import albumentations.pytorch
import albumentations.augmentations.transforms

sys.path.append(os.path.abspath('..'))

# BaseLine 코드로 주어진 dataset.py model.py, loss.py를 Import 합니다.
from dataset import MaskBaseDataset, BaseAugmentation
from model import *
from loss import create_criterion

sys.path.append('../')

import wandb

wandb.init(project="image-classification", entity="qwon135")

from albumentations.core.transforms_interface import ImageOnlyTransform
# https://dacon.io/codeshare/2360

def canny_edges(image):

    image = cv2.Canny(image,50,100)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

class Canny(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=1):        
        super(Canny, self).__init__(always_apply, p)
    
    def apply(self, img, **params):
        return canny_edges(img)


img_root = '/opt/ml/input/data/train/images'



transform = A.Compose([
    A.CenterCrop (380,380, always_apply=False, p=1.0),
#     A.HorizontalFlip(p=0.5),    
#     Canny(p = 0.5),
    A.Normalize (mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
    A.pytorch.transforms.ToTensorV2(),

])

dataset = MaskBaseDataset(img_root, transform=transform)


############ 시드 고정  ################

def seed_everything(seed):
    """
    동일한 조건으로 학습을 할 때, 동일한 결과를 얻기 위해 seed를 고정시킵니다.
    
    Args:
        seed: seed 정수값
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
seed_everything(525)


# -- settings
num_workers = 4
num_classes = 18

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

############## DataLoader #######################

def getDataloader(dataset, train_idx, valid_idx, args):
    device = torch.device("cuda" if use_cuda else "cpu")

    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set = torch.utils.data.Subset(dataset,
                                        indices=valid_idx)
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader


import pandas as pd
from torchvision.transforms import Resize, ToTensor, Normalize

from dataset import TestDataset

def oof(args):
    test_img_root = '/opt/ml/input/data/eval'
    # public, private 테스트셋이 존재하니 각각의 예측결과를 저장합니다.

    # meta 데이터와 이미지 경로를 불러옵니다.
    submission = pd.read_csv(os.path.join(test_img_root, 'info.csv'))
    image_dir = os.path.join(test_img_root, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    test_dataset = TestDataset(image_paths,transform=transform)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False
    )            
    os.makedirs(os.path.join(os.getcwd(), 'results', args.name), exist_ok=True)
    device = torch.device("cuda" if use_cuda else "cpu")

    skf = StratifiedKFold(n_splits=args.n_splits)
    gss = GroupShuffleSplit(n_splits=args.n_splits)
    groups = list(range(21161))
    
    counter = 0
    patience = args.patience
    accumulation_steps = 2
    best_val_acc = 0
    best_val_loss = np.inf
    oof_pred = None

    labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]

    # K-Fold Cross Validation과 동일하게 Train, Valid Index를 생성합니다. 

#     for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):

    # fold 랜덤생성
    for i, (train_idx, valid_idx) in enumerate(gss.split(dataset.image_paths, labels,groups=groups)):        
        train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, args)

        # -- model
        model = SwinTransformer(num_classes=18, img_size=380, window_size = 7).to(device)
#         model = models.resnet18(pretrained=True).to(device)
        torch.nn.init.xavier_uniform_(model.fc.weight)

        stdv = 0
        model.fc.bias.data.uniform_(-stdv, stdv).to(device)
        

        # -- loss 2개 섞어서 사용해보기f
        f1_loss = create_criterion('f1')
        focal_loss = create_criterion('focal')
                
        optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=5e-4
            )
#         scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=args.lr)
#         scheduler = CyclicLR(optimizer, base_lr=args.lr, max_lr=0.1, step_size_up=50, step_size_down=100, mode='triangular')
    
        # -- logging
        logger = SummaryWriter(log_dir=f"results/cv{i+1}")
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = f1_loss(outs, labels)*0.5 + focal_loss(outs, labels)*0.5

                loss.backward()

                 # -- Gradient Accumulation
                if (idx+1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.train_log_interval == 0:
                    train_loss = loss_value / args.train_log_interval
                    train_acc = matches / args.batch_size / args.train_log_interval
                    current_lr = scheduler.get_last_lr()
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                        f"현재시간 : {datetime.now()}"
                    )

                    loss_value = 0
                    matches = 0
                wandb.log({"confusion_mat": wandb.plot.confusion_matrix(preds=preds,y_true=labels)})

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...",f"현재시간 : {datetime.now()}")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs).to(device)
                    preds = torch.argmax(outs, dim=-1).to(device)

                    loss_item = (f1_loss(outs, labels)*0.5 + focal_loss(outs, labels)*0.5).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(valid_idx)

                # Callback1: validation accuracy가 향상될수록 모델을 저장합니다.
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if val_acc > best_val_acc:
                    print("New best model for val accuracy! saving the model..",f"현재시간 : {datetime.now()}")
                    torch.save(model.state_dict(), f"results/{args.name}/{epoch:03}_accuracy_{val_acc:4.2%}.ckpt")
                    best_val_acc = val_acc
                    counter = 0
                else:
                    counter += 1
                # Callback2: patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
                if counter > patience:
                    print("Early Stopping...",f"현재시간 : {datetime.now()}")
                    counter = 0
                    break

                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                    f"현재시간 : {datetime.now()}"
                    f"현재 counter : {counter}입니다. {args.patience}가 되면 종료"
                )

        # 각 fold에서 생성된 모델을 사용해 Test 데이터를 예측합니다. 
        all_predictions = []
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)

                # Test Time Augmentation
                pred = model(images) / 2 # 원본 이미지를 예측하고
                pred += model(torch.flip(images, dims=(-1,))) / 2 # horizontal_flip으로 뒤집어 예측합니다. 
                all_predictions.extend(pred.cpu().numpy())

            fold_pred = np.array(all_predictions)
            
        # 확률 값으로 앙상블을 진행하기 때문에 'k'개로 나누어줍니다.
        if oof_pred is None:
            oof_pred = fold_pred / args.n_splits
        else:
            oof_pred += fold_pred / args.n_splits
        
        counter = 0
                
    submission['ans'] = np.argmax(oof_pred, axis=1)
    submission.to_csv(os.path.join(test_img_root, 'submission.csv'), index=False)
    print('test inference is done!',f"현재시간 : {datetime.now()}")
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--n_splits', type=int, default=5, help='n_splits (default: 5)')

    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-5)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler deacy step (default: 10)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: focal)')
    parser.add_argument('--train_log_interval', type=str, default=20, help='(default: 20)')
    parser.add_argument('--name', type=str,default='02_model_Ensemble', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 150)')
    parser.add_argument('--patience', type=int, default=100, help='number of patience to train (default: 300)')

    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    
    oof(args)
    
