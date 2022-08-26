import os
import glob
import wandb
from utils.metric import AverageMeter, Summary, ProgressMeter, accuracy, lira_metrics
from tqdm import tqdm
import time
import torch
from torch import nn


class EarlyStopPatience(nn.Module):
    def __init__(self, patience=10):
        self.patience = patience
        self.count = 0
        self.best_score = None
        self.bool_early_stop = False

    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        # accumulate counting if the score is not better than the best score
        elif score < self.best_score:
            self.count += 1
            if self.count >= self.patience:
                self.bool_early_stop = True
                print("Early stopping")
        # renew count and best score if the maximum score is achieved
        elif score >= self.best_score:
            self.best_score = score
            self.count = 0
        return self.bool_early_stop


def validate(val_loader, model, criterion, device):
    # Reference: https://github.com/pytorch/examples/blob/00ea159a99f5cb3f3301a9bf0baa1a5089c7e217/imagenet/main.py#L313-L353
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4f", Summary.AVERAGE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    true_positive = AverageMeter("True Positive", ":6.2f", Summary.AVERAGE)
    false_positive = AverageMeter("False Positive", ":6.2f", Summary.AVERAGE)
    precision_rate = AverageMeter("Precision", ":6.2f", Summary.AVERAGE)

    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            labels = labels.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure LIRA metrics
            output = output.argmax(dim=-1)
            tp_rate, fp_rate, precision_score = lira_metrics(
                output.cpu().numpy(), labels.cpu().numpy()
            )
            true_positive.update(tp_rate, images.size(0))
            false_positive.update(fp_rate, images.size(0))
            precision_rate.update(precision_score, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        progress.display_summary()

    return (
        losses.avg,
        top1.avg,
        top5.avg,
        true_positive.avg,
        false_positive.avg,
        precision_rate.avg,
    )


def train(
    CFG,
    model,
    train_loader,
    valid_loader,
    optimizer,
    save_path,
    shadow_number,
    scheduler=None,
    criterion=None,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    early_stop_acc1 = EarlyStopPatience(patience=CFG.early_stop_patience)
    best_valid_acc = 0
    best_valid_loss = 10

    for epoch in range(CFG.num_epochs):
        # Train Code Reference: https://github.com/pytorch/examples/blob/00ea159a99f5cb3f3301a9bf0baa1a5089c7e217/imagenet/main.py#L266-L310
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")

        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch),
        )

        end = time.time()
        for iter, (images, labels) in enumerate(tqdm(train_loader)):
            # initialize gradients
            optimizer.zero_grad()

            # assign images and labels to the device
            # images, labels = images.type(torch.FloatTensor).to(device), labels.to(device)
            images, labels = images.to(device), labels.to(device)

            # switch to train mode
            model.train()

            # compute output
            output = model(images)
            loss = criterion(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if iter % CFG.logging_steps == 0:
                progress.display(iter)  # display train status
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": losses.avg,
                        "train/top5_accuracy": top5.avg,
                        "train/top1_accuracy": top1.avg,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

        # Validate on each epoch
        print("Epoch Finished... Validating")
        (
            valid_loss,
            valid_acc1,
            valid_acc5,
            val_true_positive,
            val_false_positive,
            val_precision,
        ) = validate(valid_loader, model, criterion, device)
        wandb.log(
            {
                "valid/loss": valid_loss,
                "valid/top5_accuracy": valid_acc5,
                "valid/top1_accuracy": valid_acc1,
                "valid/true_positive": val_true_positive,
                "valid/false_positive": val_false_positive,
                "valid/precision": val_precision,
            }
        )

        if valid_acc5 > best_valid_acc:
            # find previous checkpoint that contains shadow_{shadow_number}
            # if found, delete it
            if shadow_number >= 0:
                previous_checkpoints = glob.glob(
                    os.path.join(save_path, f"shadow_{shadow_number}_loss_*_acc5_*.ckpt")
                )
                for checkpoint in previous_checkpoints:
                    os.remove(checkpoint)

                print("New valid model for val accuracy! saving the model...")
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_path,
                        f"shadow_{shadow_number}_loss_{valid_loss:4.2}_acc5_{valid_acc5}.ckpt",
                    ),
                )
            elif shadow_number < 0:
                previous_checkpoints = glob.glob(
                    os.path.join(save_path, f"target_loss_*_acc5_*.ckpt")
                )
                for checkpoint in previous_checkpoints:
                    os.remove(checkpoint)

                print("New valid model for val accuracy! saving the model...")
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_path, f"target_loss_{valid_loss:4.2}_acc5_{valid_acc5}.ckpt",
                    ),
                )
            best_valid_acc = valid_acc5
            best_valid_loss = valid_loss
            wandb.log({"best_valid_top5_acc": best_valid_acc})

        # early stop based on validation top1 accuracy patience
        if early_stop_acc1(valid_acc1):
            break

        # early stop based on validation accuracy goal
        if CFG.val_acc_goal > 0:
            if best_valid_acc >= CFG.val_acc_goal:
                print("Early stopping...")
                break

    # load best model
    print("Loading best model...")
    print(f"Best valid loss: {best_valid_loss:4.2}")
    print(f"Best valid top5 accuracy: {best_valid_acc:4.2}")
    model.load_state_dict(
        torch.load(
            os.path.join(
                save_path,
                f"shadow_{shadow_number}_loss_{best_valid_loss:4.2}_acc5_{best_valid_acc}.ckpt",
            )
        )
    )
    return model

