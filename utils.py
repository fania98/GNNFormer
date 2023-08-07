import os
import os.path as osp
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lovasz_losses as L

class Logger:
    def __init__(self, log_dir, local_rank=0):
        self.log_dir = log_dir
        self.local_rank = local_rank
        if local_rank == 0:
            if not osp.exists(log_dir):
                os.makedirs(log_dir)
            self.log_file = osp.join(log_dir, 'log_2_classification.txt')
            self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, content):
        if self.local_rank == 0:
            with open(self.log_file, 'a+') as f:
                f.write(content + '\n')

    def add_scalar(self, name, value, iter):
        if self.local_rank == 0:
            self.writer.add_scalar(name, value, iter)

    def close(self):
        if self.local_rank == 0:
            self.writer.close()

class AccMeter:
    def __init__(self):
        self.total = 0
        self.correct = 0

    def add(self, pred, gt):
        """
        :param pred: [batch_size]
        :param gt: [batch_size]
        :return:
        """
        pred = pred.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        self.total += pred.shape[0]
        self.correct += (pred == gt).sum()

    def value(self):
        return self.correct / self.total


class MIoUMeter:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.union = [0] * num_classes
        self.intersection = [0] * num_classes

    def add(self, pred, gt):
        """
        :param pred: [batch_size, H, W]
        :param gt: [batch_size, H, W]
        :return:None
        """
        pred = pred.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        assert pred.shape == gt.shape
        batch = pred.shape[0]
        for b in range(batch):
            pred_temp = pred[b]
            gt_temp = gt[b]
            for i in range(self.num_classes):
                match = (pred_temp == i).astype(np.float32) + (gt_temp == i).astype(np.float32)
                it = (match == 2).astype(np.float32).sum()
                un = (match > 0).astype(np.float32).sum()
                self.intersection[i] += it
                self.union[i] += un

    def value(self):
        iou = []
        for i in range(self.num_classes):
            if self.union[i] == 0:
                iou.append(0)
            else:
                iou.append(self.intersection[i] / self.union[i])
        return np.array(iou)


def binary_dice(predict, label):
    """
    :param predict: [batch_size, *]
    :param label: [batch_size, *]
    :return: dice
    """
    assert predict.size() == label.size()
    batch_size = predict.size()[0]
    predict = predict.float().view(batch_size, -1)
    label = label.float().view(batch_size, -1)
    dice = 0
    for i in range(batch_size):
        intersection = predict[i] * label[i]
        dice += (2 * intersection.sum() + 1) / (predict[i].sum() + label[i].sum() + 1)
    return dice / batch_size


def one_hot(src, num_classes=2):
    """
    :param src: [batch_size, *]
    :param num_classes:
    :return:[batch_size, num_classes, *]
    """
    src = src.unsqueeze(1)
    dst = torch.zeros_like(src)
    dst = torch.cat([dst for _ in range(num_classes)], 1)
    dst.scatter_(dim=1, index=src.long(), value=1)
    return dst


class DiceLoss(nn.Module):
    def __init__(self, num_classes, weight=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = None
        if weight is not None:
            self.weight = weight.cuda()

    def forward(self, predict, label, label_softmax=False, need_softmax=False):
        """
        :param predict:[batch, num_class, *]
        :param label: [batch, num_class, *]
        :param label_softmax: whether label should be softmax
        :return: dice loss
        """
        label = label.float()
        if need_softmax:
            predict = F.softmax(predict, 1)
        if len(label.size()) != len(predict.size()):
            label = one_hot(label, self.num_classes).float()
        if label_softmax:
            label = F.softmax(label, dim=1)
        dice = 0
        for i in range(self.num_classes):
            if self.weight is None:
                dice += (1 - binary_dice(predict[:, i], label[:, i]))
            else:
                dice += self.weight[i] * (1 - binary_dice(predict[:, i], label[:, i]))
        dice /= self.num_classes
        return dice


class LovaszLoss(nn.Module):
    def __init__(self, num_classes, weight=None, classes='present', ignore=None):
        super(LovaszLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.classes = classes
        self.ignore = ignore

    def forward(self, predict, label, need_softmax=True):
        """
        predict: [batch, num_class, *]
        label: [batch, *]
        need_softmax: whether the predict needs to softmax
        """
        label = label.long()
        if need_softmax:
            predict = F.softmax(predict, 1)
        if self.weight is None:
            loss = L.lovasz_softmax(predict, label, classes=self.classes, ignore=self.ignore)
        else:
            loss = L.lovasz_softmax_weight(predict, label, weight=self.weight)
        return loss


def accuracy(output, target, topk=(1,), output_has_class_ids=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # if not output_has_class_ids:
    #     output = torch.Tensor(output)
    # else:
    #     output = torch.LongTensor(output)
    # target = torch.LongTensor(target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.shape[0]
        if not output_has_class_ids:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
        else:
            pred = output[:, :maxk].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res