import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from my_dataset import Data_myself
from model import swin_base_patch4_window7_224 as create_model
from utils import evaluate
from calculate_utils import compute_eer, compute_auc, perf_measure
from tqdm import tqdm
import sys
import numpy as np
import numpy

@torch.no_grad()
def test(model, data_loader, device, epoch, dev_eer):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    labelList = []
    predList = []
    soft_predList = []
    sample_num = 0
    attack_confidence = []
    labelnumList = []
    prenumlist = []
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, HSVimg, Ycbcrimg, labels, map_label = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        softpred = pred.softmax(dim=1)
        livepred = softpred[0][1]
        pred_classes = softpred.argmax(dim=1)
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        labelList.append(labels.data.cpu().numpy())
        labelnumList.append(labels.item())
        predList.append(pred_classes.data.cuda())
        soft_predList = numpy.append(soft_predList,softpred.data[0][1].cpu().numpy())
        soft_predList = numpy.array(soft_predList)
        prenumlist.append(round(livepred.item(),2))
        attack_confidence.append(softpred.data[0][1].cpu())
    labelList = np.concatenate(labelList)
    auc = compute_auc(labelnumList, prenumlist)
    eer = compute_eer(labelList, soft_predList)
    bpcer, apcer, HTER = perf_measure(labelnumList, prenumlist, dev_eer)
    return eer, auc, (apcer+bpcer)/2

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    model = create_model(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    test_dataset = Data_myself(listroot=args.test_list_path, transform=data_transform["test"])
    val_dataset = Data_myself(listroot=args.dev_list_path, transform=data_transform["val"])
    batch_size = args.batch_size
    val_loader = DataLoader(val_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        pin_memory=False,
                                        num_workers=0,
                                        )
    test_loader = DataLoader(test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        pin_memory=False,
                                        num_workers=0,
                                        )
    val_loss, val_acc, val_threshold = evaluate(model=model,
                                                data_loader=val_loader,
                                                device=device,
                                                epoch=1)
    eer, auc, acer = test(model=model, data_loader=test_loader, device=device, epoch=1, dev_eer=val_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--dev_list_path', type=str, default='')
    parser.add_argument('--test_list_path', type=str, default='')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
