import argparse
import multiprocessing
import os
from importlib import import_module
from torchvision import models

import pandas as pd
import torch
from torch.utils.data import DataLoader
# from efficientnet_pytorch import EfficientNet
import albumentations as A
import albumentations.pytorch
import albumentations.augmentations.transforms


from dataset import TestDataset, MaskBaseDataset

transform = A.Compose([
    A.Resize(512,384), #사진 크기가 같아서 필요 X,
    A.CenterCrop (380,380, always_apply=False, p=1.0),
#     A.HorizontalFlip(p=0.5),    
#     Canny(p = 0.5),
    A.Normalize (mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
    A.pytorch.transforms.ToTensorV2(),

])

def load_model(saved_model, num_classes, device):
    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)    
    

#     model = models.shufflenet_v2_x1_0(pretrained=True).to(device)
#     model.fc = torch.nn.Linear(in_features = 1024, out_features = 18, bias = True)

    model = models.resnet18(pretrained=True).to(device)
    model.fc = torch.nn.Linear(in_features = 512, out_features = 18, bias = True)

#     torch.nn.init.xavier_uniform_(model.fc.weight)
#     stdv = 1/np.sqrt(512)v2/results/02_model_Ensemble/
#     model.fc.bias.data.uniform_(-stdv, stdv)
    model_path = os.path.join(saved_model, '024_accuracy_85.92%.ckpt')
#     model_path = os.path.join(saved_model, 'best.pth')    
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple,  default=[256, 192], help='resize size for image when you trained (default: (128, 96))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
#     parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp3'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './results/02_model_Ensemble'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
