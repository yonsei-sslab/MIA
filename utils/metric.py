import torch
from enum import Enum
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def lira_metrics(output, target, bool_average=True, num_classes=10):

    # Source of tp, fp rate: https://dev.to/overrideveloper/understanding-the-confusion-matrix-264i
    sum_true_postive_rate, sum_false_positive_rate = 0, 0

    # output as 2 x 2 matrix
    multi_cm_outputs = multilabel_confusion_matrix(target, output)

    for cm_output in multi_cm_outputs:
        (
            tn,
            fp,
            fn,
            tp,
        ) = (
            cm_output.ravel()
        )  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        sum_true_postive_rate += tp / (tp + fn)
        sum_false_positive_rate += fp / (fp + tn)

    # get the precision rate = TP / (FP + TP)
    precision_rate = precision_score(
        target,
        output,
        average="micro",  # Source of micro vs macro: https://androidkt.com/micro-macro-averages-for-imbalance-multiclass-classification/#:~:text=The%20difference%20between%20macro%20and,result%20in%20the%20same%20score.
    )

    # HOW SHALL I MAKE FPR:TPR Ratio according to threshold(=ROC)?
    # true_postive_rate, false_positive_rate, thresholds = roc_curve(target, output, pos_label=class_num)

    if bool_average:
        tpr_class_avg = sum_true_postive_rate / num_classes
        fpr_class_avg = sum_false_positive_rate / num_classes
        return tpr_class_avg, fpr_class_avg, precision_rate
    else:
        return sum_true_postive_rate, sum_false_positive_rate, precision_rate


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

