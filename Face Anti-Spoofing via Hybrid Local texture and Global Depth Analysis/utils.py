import sys
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve
from calculate_utils import get_err_threhold

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    loss_function1 = torch.nn.MSELoss().cuda()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, HSVimg, Ycbcrimg, labels, map_label = data
        sample_num += images.shape[0]
        pred, map_x, _ = model(images.to(device))
        map_label = map_label.to(device)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        spoof_loss = loss_function(pred, labels.to(device))
        absolute_loss = loss_function1(map_x, map_label)
        loss = spoof_loss + absolute_loss
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    labelnumList = []
    prenumlist = []
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, HSVimg, Ycbcrimg, labels, map_label = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        softpred = pred.softmax(dim=1)
        livepred = softpred[0][1]
        labelnumList.append(labels.item())
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        prenumlist.append(round(livepred.item(), 2))
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    fpr, tpr, threshold = roc_curve(labelnumList, prenumlist, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, val_threshold
