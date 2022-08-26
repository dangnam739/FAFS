from ast import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import pdb

from ..backbones.backbone_builder import build_backbone
from ..decoder.decoder_builder import build_decoder
from ..predictor.predictor_builder import build_predictor
from ..losses.loss_builder import build_loss
from ..discriminator.discriminator_builder import build_discriminator

class UDASegmentor(nn.Module):
    """
    unsupervised domain adaptation segmentor
    """
    def __init__(self, cfg):
        super(UDASegmentor, self).__init__()

        self.backbone = build_backbone(cfg)
        self.decoder = build_decoder(cfg, self.backbone.out_channels)
        self.predictor = build_predictor(cfg, self.decoder.out_channels)
        self.loss = build_loss(cfg)
        self.cfg = cfg

        self.discriminators = nn.Sequential()
        for name, D in build_discriminator(
            cfg, 
            self.backbone.out_channels, 
            self.decoder.out_channels, 
            self.predictor.out_channels
            ).items():
            self.discriminators.add_module(name, D)
        
        self.discriminator_loss = build_loss(cfg, is_discriminator=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, source, target=None, source_label=None, source_label_onehot=None, target_label=None, target_label_onehot=None):
        # source domain
        if not self.training or self.cfg.MODEL.DISCRIMINATOR.WEIGHT or self.cfg.DATASET.TARGET.SOURCE_LOSS_WEIGHT>0:
            s_size = source.shape[-2:]
            s_features = self.backbone(source)
            s_decoder_out = self.decoder(s_features)

            if isinstance(s_decoder_out, tuple):
                s_decoder_fm = s_decoder_out[1]
                s_decoder_out = s_decoder_out[0]
            else:
                s_decoder_fm = s_decoder_out            
            if not isinstance(s_decoder_fm, list):
                s_decoder_fm = [s_decoder_fm]
            s_logits = self.predictor(s_decoder_out, source)
            del s_decoder_out, 

        if self.training:
            # target domain
            t_size = target.shape[-2:]
            t_features = self.backbone(target)
            t_decoder_out = self.decoder(t_features)
            if isinstance(t_decoder_out, tuple):
                t_decoder_fm = t_decoder_out[1]
                t_decoder_out = t_decoder_out[0]
            else:
                t_decoder_fm = t_decoder_out
            
            if not isinstance(t_decoder_fm, list):
                t_decoder_fm = [t_decoder_fm]
            t_logits = self.predictor(t_decoder_out, target)
            t_logits_softmax = self.softmax(t_logits)
            del t_decoder_out
            
            # defaut reg_weight
            if self.cfg.MODEL.DISCRIMINATOR.LAMBDA_ENTROPY_WEIGHT or self.cfg.MODEL.DISCRIMINATOR.LAMBDA_KLDREG_WEIGHT:
                reg_val_matrix = torch.ones_like(target_label).type_as(t_logits)
                reg_val_matrix[target_label==255]=0
                reg_val_matrix = reg_val_matrix.unsqueeze(dim=1)
                reg_ignore_matrix = 1 - reg_val_matrix
                reg_weight = torch.ones_like(t_logits)
                reg_weight_val = reg_weight * reg_val_matrix
                reg_weight_ignore = reg_weight * reg_ignore_matrix
                del reg_ignore_matrix, reg_weight, reg_val_matrix
            
            losses = {}

            # update discriminators
            for name, D in self.discriminators.named_children():
                
                if "Pixel" in name:
                    if isinstance(s_features, tuple) or isinstance(s_features, list):
                        s_D_logits = D(s_features[-1].detach(), s_size)
                        t_D_logits = D(t_features[-1].detach(), t_size) 
                    else:
                        s_D_logits = D(s_features.detach(), s_size)
                        t_D_logits = D(t_features.detach(), t_size)                   
                    
                    temperature = 1.8
                    s_soft_label = self.softmax(s_logits.div(temperature)).detach()
                    s_soft_label[s_soft_label>0.9] = 0.9
                    t_soft_label = self.softmax(t_logits.div(temperature)).detach()
                    t_soft_label[t_soft_label>0.9] = 0.9

                    is_source = torch.cat((s_soft_label, torch.zeros_like(s_soft_label)), dim=1)
                    is_target = torch.cat((torch.zeros_like(t_soft_label), t_soft_label), dim=1)

                    D_loss_src = 0.5*soft_label_cross_entropy(s_D_logits, is_source)
                    D_loss_tgt = 0.5*soft_label_cross_entropy(t_D_logits, is_target)

                    discriminator_loss = (D_loss_src + D_loss_tgt)

                    losses.update({'D_src_loss': D_loss_src})
                    losses.update({'D_tgt_loss': D_loss_tgt})
                    losses.update({'D_' + name + '_loss': discriminator_loss})

                elif "Origin" in name:
                    s_D_logits = D(self.softmax(s_logits).detach())
                    t_D_logits = D(self.softmax(t_logits).detach())
                                
                    if isinstance(s_D_logits, tuple):
                        is_source = torch.zeros_like(s_D_logits[0]).cuda()
                        is_target = torch.ones_like(t_D_logits[0]).cuda()
                        for j, (s, t) in enumerate(zip(s_D_logits, t_D_logits)):
                            discriminator_loss = (self.discriminator_loss(s, is_source) + 
                                            self.discriminator_loss(t, is_target))/2
                            losses.update({'D_' + name  + '_loss'+ str(j): discriminator_loss})
                    else:
                        is_source = torch.zeros_like(s_D_logits).cuda()
                        is_target = torch.ones_like(t_D_logits).cuda()
                        discriminator_loss = (self.discriminator_loss(s_D_logits, is_source) + 
                                    self.discriminator_loss(t_D_logits, is_target))/2
                        losses.update({'D_' + name + '_loss': discriminator_loss})
                else:
                    break

            # adv_losses
            adv_losses = []
            for i, value in enumerate(self.discriminators.named_children()):
                name, D = value

                if "Pixel" in name:
                    t_D_logits = D(t_features, t_size)
                    is_source = torch.cat((t_soft_label, torch.zeros_like(t_soft_label)), dim=1).cuda()
                    adv_loss = 0.001*soft_label_cross_entropy(t_D_logits, is_source)
                    adv_losses.append(adv_loss)                  

                elif "Origin" in name:
                    t_D_logits = D(self.softmax(t_logits))
                               
                    t_size_sample = t_D_logits[0] if isinstance(t_D_logits, tuple) else t_D_logits
                    if isinstance(t_D_logits, tuple):
                        for j, t in enumerate(t_D_logits):
                            discriminator_loss = self.discriminator_loss(t, is_source)
                            adv_losses.append(self.cfg.MODEL.DISCRIMINATOR.WEIGHT[i] * discriminator_loss)
                    else:
                        is_source = torch.zeros_like(t_size_sample).cuda()
                        discriminator_loss = self.discriminator_loss(t_D_logits, is_source)
                        adv_losses.append(self.cfg.MODEL.DISCRIMINATOR.WEIGHT[i] * discriminator_loss)
                else:
                    break
            
            # update seg loss
            if self.cfg.DATASET.TARGET.SOURCE_LOSS_WEIGHT>0:
                mask_loss = self.cfg.DATASET.TARGET.SOURCE_LOSS_WEIGHT * self.loss(s_logits, source_label) + sum(adv_losses)
                
                if self.cfg.DATASET.TARGET.ENT_LOSS_WEIGHT>0:
                    ita = self.cfg.DATASET.TARGET.ENT_ITA
                    entW = self.cfg.DATASET.TARGET.ENT_LOSS_WEIGHT

                    # compute ent loss
                    P = F.softmax(t_logits, dim=1)        # [B, 19, H, W]
                    logP = F.log_softmax(t_logits, dim=1) # [B, 19, H, W]
                    PlogP = P * logP               # [B, 19, H, W]
                    ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W]
                    ent = ent / 2.9444         # chanage when classes is not 19
                    # compute robust entropy
                    ent = ent ** 2.0 + 1e-8
                    ent = ent ** ita
                    loss_ent = ent.mean()

                    mask_loss += entW*loss_ent

                loss = {"mask_loss": mask_loss}
                losses.update(loss)

            
            
            # pseudo label target seg loss
            if target_label is not None:
                target_seg_loss = self.cfg.DATASET.TARGET.PSEUDO_LOSS_WEIGHT * self.loss(t_logits, target_label)
                losses.update({"target_seg_loss": target_seg_loss})

            # entropy reg
            if self.cfg.MODEL.DISCRIMINATOR.LAMBDA_ENTROPY_WEIGHT > 0:
                entropy_reg_loss = entropyloss(t_logits, reg_weight_ignore)
                entropy_reg_loss =  entropy_reg_loss * self.cfg.MODEL.DISCRIMINATOR.LAMBDA_ENTROPY_WEIGHT
                losses.update({"entropy_reg_loss": entropy_reg_loss})

            # kld reg
            if self.cfg.MODEL.DISCRIMINATOR.LAMBDA_KLDREG_WEIGHT > 0:
                kld_reg_loss = kldloss(t_logits, reg_weight_val)
                kld_reg_loss =  kld_reg_loss * self.cfg.MODEL.DISCRIMINATOR.LAMBDA_KLDREG_WEIGHT
                losses.update({"kld_reg_loss": kld_reg_loss})

            return losses
        
        return s_logits

def entropyloss(logits, weight=None):
    """
    logits:     N * C * H * W 
    weight:     N * 1 * H * W
    """
    val_num = weight[weight>0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    num_classed = logits.size()[1]
    entropy = -torch.softmax(logits, dim=1) * weight * logits_log_softmax
    entropy_reg = torch.sum(entropy) / val_num
    return entropy_reg

def kldloss(logits, weight):
    """
    logits:     N * C * H * W 
    weight:     N * 1 * H * W
    """
    val_num = weight[weight>0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    num_classes = logits.size()[1]
    kld = - 1/num_classes * weight * logits_log_softmax
    kld_reg = torch.sum(kld) / val_num
    return kld_reg

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))