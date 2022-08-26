import torch.nn as nn
from ..backbones.backbone_builder import build_backbone
from ..decoder.decoder_builder import build_decoder
from ..predictor.predictor_builder import build_predictor
from ..losses.loss_builder import build_loss
import torch.nn.functional as F

class GeneralizedSegmentor(nn.Module):
    '''
    encoder + decoder
    1) Deeplab v2
    '''
    def __init__(self, cfg):
        super(GeneralizedSegmentor, self).__init__()

        self.backbone = build_backbone(cfg)
        self.decoder = build_decoder(cfg, self.backbone.out_channels)
        self.predictor = build_predictor(cfg, self.decoder.out_channels)
        self.loss = build_loss(cfg)
        self.cfg = cfg

    def forward(self, images, labels=None) :
        features = self.backbone(images)
        decoder_out = self.decoder(features)
        logits = self.predictor(decoder_out, images)

        if self.training:
            losses = {}

            # calculate cross entropy 
            if labels is not None:
                loss_ce = self.loss(logits, labels)
                losses.update({"loss_ce": loss_ce})

            # calculate loss_ent - chaerbonier penatly
            if self.cfg.DATASET.TARGET.ENT_LOSS_WEIGHT>0:
                ita = self.cfg.DATASET.TARGET.ENT_ITA
                P = F.softmax(logits, dim=1)        # [B, 19, H, W]
                logP = F.log_softmax(logits, dim=1) # [B, 19, H, W]
                PlogP = P * logP               # [B, 19, H, W]
                ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W]
                ent = ent / 2.9444         # chanage when classes is not 19
                # compute robust entropy
                ent = ent ** 2.0 + 1e-8
                ent = ent ** ita
                loss_ent = ent.mean()            
                losses.update({"loss_ent": loss_ent})

            return losses
        
        return logits