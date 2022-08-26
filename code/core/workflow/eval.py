import torch
from torch import tensor
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler, RandomSampler
from torch.autograd import Variable

from ..datasets.loader.dataset import BaseDataset
from ..datasets.loader.gtav_dataset import GTAVDataset
from ..datasets.loader.cityscapes_dataset import CityscapesDataset
from ..datasets.loader.bdd_dataset import BDDDataset

from ..datasets.metrics.miou import intersectionAndUnionGPU
from ..datasets.metrics.acc import acc, acc_with_hist
from ..models.losses.ranger import Ranger
from ..models.losses.cos_annealing_with_restart import CosineAnnealingLR_with_Restart
from ..models.registry import DATASET

import os
import time
import numpy as np
import tqdm
import pdb

trainid2name = {
        "0": "road",
        "1": "sidewalk",
        "2": "building",
        "3": "wall",
        "4": "fence",
        "5": "pole",
        "6": "light",
        "7": "sign",
        "8": "vegetation",
        "9": "terrain",
        "10": "sky",
        "11": "person",
        "12": "rider",
        "13": "car",
        "14": "truck",
        "15": "bus",
        "16": "train",
        "17": "motorcycle",
        "18": "bicycle"
    }

def eval_net(net, cfg, gpu, logger):
    
    # train dataset
    result = []
    phase_name = (cfg.TRAIN.RESUME_FROM).split('/')[-2]
    early_stopping = cfg.TRAIN.EARLY_STOPPING
    anns = cfg.DATASET.ANNS
    image_dir = cfg.DATASET.IMAGEDIR
    use_aug = cfg.DATASET.USE_AUG
    test_sizes = cfg.TEST.RESIZE_SIZE
    bs = cfg.TEST.BATCH_SIZE
    num_work = cfg.TEST.NUM_WORKER
    use_flip = cfg.TEST.USE_FLIP
    use_mst = cfg.TEST.USE_MST
    visualize = cfg.TEST.VISUALIZE
    extract_confusion_matrix = cfg.TEST.EXTRACT_CONFUSION_MATRIX
    base_visualize_dir = '/home/nav/namkd/rnd-domain-adaptation/results/visualize/test_visualize/sl_fft_update_pl_0210'

    if visualize:
        visualize_dir = os.path.join(base_visualize_dir, phase_name)
        if not os.path.exists(visualize_dir):
            try:
                os.makedirs(visualize_dir)
            except Exception:
                pass 
    
    # init net
    dist.init_process_group(
        backend='nccl', 
        init_method='tcp://127.0.0.1:6784', 
        world_size=cfg.TRAIN.N_PROC_PER_NODE,
        rank=gpu
        )
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:{}".format(gpu))
    net.to(device)
    
    # val dataset  
    val_anns = cfg.DATASET.VAL.ANNS
    val_image_dir = cfg.DATASET.VAL.IMAGEDIR
    val = DATASET[cfg.DATASET.VAL.TYPE](val_anns, val_image_dir)
    val_sampler = DistributedSampler(val, num_replicas=cfg.TEST.N_PROC_PER_NODE, rank=gpu)
    val_data = DataLoader(val, bs, num_workers=num_work, sampler=val_sampler)

    if gpu == 0:
        logger.info('Eval Size: {}'.format(test_sizes))
        logger.info('Use Flip: {}'.format(use_flip))
        logger.info('Use MST: {}'.format(use_mst))
    
    with torch.no_grad():
        net.eval()
        intersection_sum = 0
        union_sum = 0
        n_class = cfg.MODEL.PREDICTOR.NUM_CLASSES

        if extract_confusion_matrix:
            confusion_matrix = np.zeros((n_class, n_class), dtype=np.int64)
        
        for test_size in test_sizes:
            intersection_sum = 0
            union_sum = 0

            if use_mst:
                scales = test_sizes
            else:
                logger.info("Test size: {}".format(test_size))
                scales = [test_size]

            if gpu == 0:
                pbar = tqdm.tqdm(total=len(val_data))

            for i, b in enumerate(val_data):
                if gpu == 0 :
                    pbar.update(1)
                images = b[0].cuda(non_blocking=True)
                labels = b[1].type(torch.LongTensor).cuda(non_blocking=True)
                names = b[2]

                pred_result = []
                for scale in scales:
                    tmp_images = F.interpolate(images, scale[::-1], mode='bilinear', align_corners=True)
                    logits = F.softmax(net(tmp_images), dim=1)

                    if use_flip:
                        flip_logits = F.softmax(net(torch.flip(tmp_images, dims=[3])), dim=1)
                        logits += torch.flip(flip_logits, dims=[3])

                    logits = F.interpolate(logits, labels.size()[1:], mode='bilinear', align_corners=True)
                    pred_result.append(logits)
                result = sum(pred_result)

                label_pred = result.max(dim=1)[1]

                if test_size==[2048,1024]:
                    for _i, name in enumerate(names):
                        true_label = labels[_i]
                        pred_label = label_pred[_i]
                        true_label = true_label.data.cpu().numpy()
                        pred_label = pred_label.data.cpu().numpy()

                        if visualize:
                            name = os.path.splitext(os.path.basename(name))[0]
                            label_pred_name = name + '_mask_pred.png' 
                            label_pred_path = os.path.join(visualize_dir, label_pred_name)  
                            colorize_mask(pred_label).save(label_pred_path)
                        
                        #calculate confusion matrix
                        if extract_confusion_matrix:
                            gt = true_label.flatten()
                            pd = pred_label.flatten()

                            stacked = np.stack((gt, pd), axis=-1)
                            for p in stacked:
                                tl, pl = p
                                if tl != 255:
                                    confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1        

                intersection, union = intersectionAndUnionGPU(label_pred, labels, n_class)
                intersection_sum += intersection
                union_sum += union

            if extract_confusion_matrix:
                logger.info("Confusion Matrix: \n" + print_confusion_matrix(confusion_matrix))
                save_img = '/home/nav/namkd/rnd-domain-adaptation/results/visualize/loss_miou/confusion_matrix.png'
                plot_confusion_matrix(confusion_matrix, list(trainid2name.values()), save_img)

            if gpu == 0:
                pbar.close()
                

            dist.all_reduce(intersection_sum), dist.all_reduce(union_sum)
            intersection_sum = intersection_sum.cpu().numpy()
            union_sum = union_sum.cpu().numpy()

            if gpu == 0:
                iu = intersection_sum / (union_sum + 1e-10)
                mean_iu = np.mean(iu)
                logger.info('Testing Results: \nmiou: {:.4f}\n'.format(mean_iu) + print_iou_list(iu))
            
            if use_mst:
                break


def print_iou_list(iou_list):
    res = ''
    for i, iou in enumerate(iou_list):
        res += '{}: {:.4f}\n'.format(i, iou)
    return res

def print_confusion_matrix(cm):
    res = '['
    n_class = cm.shape[0]
    for i in range(n_class):
        res += '['
        for j in range(n_class-1):
            res += '{:10d}, '.format(cm[i,j]) 
        res += '{:10d}], \n'.format(cm[i, n_class-1])
    res += ']\n'

    return res

def colorize_mask(mask):
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def plot_confusion_matrix(cm, class_name, save_img, normalize=True, title='Confusion matrix', cmap=plt.cm.Reds):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize = (12, 10))
    sns.heatmap(cm, annot=True, annot_kws={"fontsize": 9}, fmt=".2f", cmap="Blues")

    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(class_name))
    plt.xticks(tick_marks, class_name, rotation=60)
    plt.yticks(tick_marks, class_name, rotation=0)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_img)