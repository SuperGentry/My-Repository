import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from my_dataset import Data_myself
from model import swin_base_patch4_window7_224 as create_model
from utils import evaluate, train_one_epoch
from test import test
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    torch.backends.cudnn.enabled = False
    tb_writer = SummaryWriter()
    img_size = 224
    data_transform = {
        "train": transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.RandomResizedCrop(img_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize(int(img_size * 1.143)),
                                    transforms.CenterCrop(img_size),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test":transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize(int(img_size * 1.14)),
                                    transforms.CenterCrop(img_size),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    }
    train_dataset = Data_myself(listroot=args.train_list_path, transform=data_transform["train"])
    val_dataset = Data_myself(listroot=args.dev_list_path, transform=data_transform["val"])
    test_dataset = Data_myself(listroot=args.test_list_path, transform=data_transform["test"])
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=False,
                                            num_workers=0,
                                            )

    val_loader = DataLoader(val_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        pin_memory=False,
                                        num_workers=0,
                                        )
    test_loader = DataLoader(test_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        pin_memory=False,
                                        num_workers=0,
                                        )
    model = create_model().to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(args.epochs):
        scheduler.step()
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        val_loss, val_acc, val_threshold = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        torch.save(model.state_dict(), "./weights/ablation_model-{}.pth".format(epoch))
        eer, auc, acer = test(model=model, data_loader=test_loader, device=device, epoch=epoch, dev_eer=val_threshold)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes ', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--step_size', type=int, default=3, help='how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--train_list_path', type=str, default='')
    parser.add_argument('--dev_list_path', type=str, default='')
    parser.add_argument('--test_list_path', type=str, default='')
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
