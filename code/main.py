import torch
import argparse
import os
import torch.nn as nn
import pdb

import torch.multiprocessing as mp

from core.models.segmentors.generalized_segmentor import GeneralizedSegmentor
from core.models.segmentors.uda_segmentor import UDASegmentor
from core.models.default import cfg
from core.models.backbones import resnet, res2net, efficientnet, resnet_stodepth, mix_transformer
from core.models.decoder import deeplabv2_decoder, segformer_decoder
from core.models.predictor import base_predictor
from core.models.losses import mse_loss, bce_loss
from core.models.discriminator import base_discriminator, pixel_discriminator
from core.workflow.trainer import train_net
from core.workflow.sl_trainer import self_train_net

def main_worker(proc_idx, cfg):
    if cfg.MODEL.TYPE== "Generalized_Segmentor":
        net = GeneralizedSegmentor(cfg)
    elif cfg.MODEL.TYPE== "UDA_Segmentor" or cfg.MODEL.TYPE == "SL_UDA_Segmentor":
        net = UDASegmentor(cfg)
    else:
        raise Exception('error MODEL.TYPE {} !'.format(cfg.MODEL.TYPE))

    # resume main net
    state_dict = None
    last_cp = os.path.join(cfg.WORK_DIR, 'last_iter.pth')
    resume_cp_path = None
    if os.path.exists(last_cp):
        resume_cp_path = last_cp
    elif cfg.TRAIN.RESUME_FROM != "":
        resume_cp_path = cfg.TRAIN.RESUME_FROM
         
    if resume_cp_path:
        print(resume_cp_path)
        state_dict = torch.load(resume_cp_path, map_location=torch.device('cpu'))
    
    if state_dict:
        model_dict = net.state_dict()
        if "module." in list(state_dict.keys())[0]:
            pretrained_dict = {k[7:]: v for k, v in state_dict.items() if k[7:] in model_dict}
        else:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    
    # resume pseudo net
    state_dict = None
    if cfg.MODEL.TYPE== "Generalized_Segmentor":
        pseudo_net = GeneralizedSegmentor(cfg)
    elif cfg.MODEL.TYPE== "UDA_Segmentor" or cfg.MODEL.TYPE == "SL_UDA_Segmentor":
        pseudo_net = UDASegmentor(cfg)
    else:
        raise Exception('error MODEL.TYPE {} !'.format(cfg.MODEL.TYPE))

    if cfg.TRAIN.PSEUDO_RESUME_FROM != "":
        state_dict = torch.load(cfg.TRAIN.PSEUDO_RESUME_FROM, map_location=torch.device('cpu'))

        if "module." in list(state_dict.keys())[0]:
            # remove "module."
            pretrained_dict = {k[7:]: v for k, v in state_dict.items() if k[7:] in model_dict}
        else:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        pseudo_net.load_state_dict(model_dict)
    
    if cfg.MODEL.TYPE == "SL_UDA_Segmentor":
        self_train_net(net=net, net_pseudo=pseudo_net, cfg=cfg, gpu=proc_idx)
    else:
        train_net(net=net, cfg=cfg, gpu=proc_idx)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch core")
    parser.add_argument(
        "--config_file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--pseudo_resume_from", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default=None)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.nproc_per_node:
        cfg.TRAIN.N_PROC_PER_NODE = args.nproc_per_node
    if args.resume_from:
        cfg.TRAIN.RESUME_FROM = args.resume_from
    if args.pseudo_resume_from:
        cfg.TRAIN.PSEUDO_RESUME_FROM = args.pseudo_resume_from
    if args.work_dir:
        cfg.WORK_DIR = args.work_dir
    cfg.freeze()

    dir_cp = cfg.WORK_DIR
    if not os.path.exists(dir_cp):
        os.makedirs(dir_cp)

    mp.spawn(main_worker, nprocs=cfg.TRAIN.N_PROC_PER_NODE, args=(cfg,))
    
