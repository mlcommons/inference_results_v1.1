import csv
import torch
from collections import OrderedDict


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = self.avg = self.sum = self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ClassificationAccuracy:
    def __init__(self):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    def measure(self, output: torch.Tensor, target: torch.Tensor):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        topk = (1, 5)
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        acc1, acc5 = [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]
        self.top1.update(acc1.item(), batch_size)
        self.top5.update(acc5.item(), batch_size)

    def announce(self):
        results = OrderedDict(top1=round(self.top1.avg, 4), top5=round(self.top5.avg, 4))
        return results

    def print_result(self):
        results = self.announce()
        print(f'\t top_1 acc.: {results["top1"]}\n' f'\t top_5 acc.: {results["top5"]}\n')

    def write_result(self, path):
        results = self.announce()
        with open(path, mode="w") as cf:
            dw = csv.DictWriter(cf, fieldnames=results.keys())
            dw.writeheader()
            dw.writerow(results)
            cf.flush()
