import os
import glob
import wandb
from tqdm import tqdm
import time
import torch
from torch import nn


def make_member_nonmember(finetuned_model, trainloader, valloader, criterion, device):
    """ 
    - finetuned_model: finetuned shadow model
    - trainloader: member
    - valloader: non-member
    - criterion: loss function
    - device: cuda or cpu
    """

    member_dset = []
    non_member_dset = []

    finetuned_model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(trainloader)):
            images = images.to(device)
            labels = labels.to(device)

            # compute output
            output = finetuned_model(images)
            # softmax output
            output = torch.softmax(output, dim=1)
            output = output.cpu().detach().numpy()
            # loss = criterion(output, labels)
            # append to member_dset
            member_dset.append(output)

        for i, (images, labels) in enumerate(tqdm(valloader)):
            images = images.to(device)
            labels = labels.to(device)

            # compute output
            output = finetuned_model(images)
            output = torch.softmax(output, dim=1)
            output = output.cpu().detach().numpy()
            # loss = criterion(output, labels)
            # append to non_member_dset
            non_member_dset.append(output)

    return member_dset, non_member_dset
