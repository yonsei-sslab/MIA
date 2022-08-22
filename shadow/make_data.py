import os
import glob
import wandb
from tqdm import tqdm
import time
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])


def make_member_nonmember(finetuned_model, trainloader, testloader, criterion, device):
    """ 
    - finetuned_model: finetuned shadow model
    - trainloader: member
    - testloader: non-member
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
            prob = F.softmax(output, dim=1)  # softmax logits

            # get top inferred classes probability and append to member_dset
            top_p, top_class = prob.topk(CFG.topk_num_accessible_probs, dim=1)
            top_p = top_p.cpu().detach().numpy()  # detach from cuda
            # loss = criterion(output, labels)
            member_dset.append(top_p)

        for i, (images, labels) in enumerate(tqdm(testloader)):
            images = images.to(device)
            labels = labels.to(device)

            # compute output
            output = finetuned_model(images)
            prob = F.softmax(output, dim=1)  # softmax logits

            # get top inferred classes probability and append to member_dset
            top_p, top_class = prob.topk(CFG.topk_num_accessible_probs, dim=1)
            top_p = top_p.cpu().detach().numpy()  # detach from cuda

            # append to non_member_dset
            non_member_dset.append(top_p)

    # change into numpy array type
    member_dset, non_member_dset = np.array(member_dset), np.array(non_member_dset)

    # return as dataset row x number of accessible probabilities: (ex) 25000 x 5, 25000 x 5
    return np.concatenate(member_dset, axis=0), np.concatenate(non_member_dset, axis=0)

