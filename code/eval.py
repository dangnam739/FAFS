import torch
import argparse
import os
import pdb
import torch.multiprocessing as mp
import logging
import sys


from core.models.segmentors.generalized_segmentor import GeneralizedSegmentor
from core.models.segmentors.uda_segmentor import UDASegmentor
from core.models.default import cfg
from core.models.backbones import resnet, efficientnet, mix_transformer, resnet_stodepth, res2net
from core.models.decoder import deeplabv2_decoder, segformer_decoder
from core.models.predictor import base_predictor
from core.models.losses import mse_loss, bce_loss
from core.models.discriminator import base_discriminator, pixel_discriminator
from core.workflow.trainer import train_net

from core.workflow.eval import eval_net

def setup_logger(name, save_dir, distributed_rank):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, name + ".log")),
            logging.StreamHandler()
        ]
    )
    #Creating an object
    logger=logging.getLogger(name)
    return logger

def main_worker(proc_idx, cfg):
    model_name = (cfg.TRAIN.RESUME_FROM).split('/')[-1][:-4]
    logger = setup_logger(model_name + "_test", cfg.WORK_DIR, None)

    if cfg.MODEL.TYPE== "Generalized_Segmentor":
        net = GeneralizedSegmentor(cfg)
    elif cfg.MODEL.TYPE== "UDA_Segmentor" or cfg.MODEL.TYPE== "SL_UDA_Segmentor":
        net = UDASegmentor(cfg)
    else:
        raise Exception('error MODEL.TYPE {} !'.format(cfg.MODEL.TYPE))

    # resume main net
    state_dict = None
    last_cp = os.path.join(cfg.WORK_DIR, 'last_epoch.pth')
    resume_cp_path = None
    if cfg.TRAIN.RESUME_FROM != "":
        resume_cp_path = cfg.TRAIN.RESUME_FROM
        state_dict = torch.load(resume_cp_path, map_location=torch.device('cpu'))
    
    if state_dict:
        model_dict = net.state_dict()
        if "module." in list(state_dict.keys())[0]:
            pretrained_dict = {k[7:]: v for k, v in state_dict.items() if k[7:] in model_dict}
        else:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    if proc_idx == 0:
        logger.info('Resume from: {}'.format(resume_cp_path))
    eval_net(net=net, cfg=cfg, gpu=proc_idx, logger=logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch core")
    parser.add_argument(
        "--config_file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--resume_from")
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--work_dir", type=str, default=None)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.resume_from:
        cfg.TRAIN.RESUME_FROM = args.resume_from
    if args.work_dir:
        cfg.WORK_DIR = args.work_dir
    cfg.freeze()

    dir_cp = cfg.WORK_DIR
    if not os.path.exists(dir_cp):
        os.makedirs(dir_cp)

    mp.spawn(main_worker, nprocs=args.gpu_num, args=(cfg,))
    
    

