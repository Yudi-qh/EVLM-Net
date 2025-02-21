import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet


parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True, 
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True, 
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20, 
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()

import torch
import torch.nn.functional as F


def focal_loss(pred, mask, gamma=2.0, alpha=0.25):
    pred = torch.sigmoid(pred)
    bce_loss = F.binary_cross_entropy(pred, mask, reduction='none')
    pt = mask * pred + (1 - mask) * (1 - pred)  
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss


def generalized_iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) - inter

    iou = inter / (union + 1e-6)  
    pred_boxes = torch.stack([pred.sum(dim=(2, 3)), mask.sum(dim=(2, 3))], dim=1)
    giou = iou - (pred_boxes.max(dim=1).values - inter) / (pred_boxes.sum(dim=1) + 1e-6)  # Generalized IoU

    return 1 - giou.mean()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    wfocal = focal_loss(pred, mask)
    wbce = (weit * wfocal).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    wiou = generalized_iou_loss(pred, mask)

    return (wbce + wiou).mean()




def main(args):    
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAM2UNet(args.hiera_path)
    model.to(device)
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(args.epoch):
        for i, batch in enumerate(dataloader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            loss.backward()
            optim.step()
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
                
        scheduler.step()
        if (epoch+1) % 5 == 0 or (epoch+1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2-UNet-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(args.save_path, 'SAM2-UNet-%d.pth'% (epoch + 1)))


# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    main(args)
