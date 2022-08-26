from typing import Dict
from ..registry import DISCRIMINATOR

def build_discriminator(cfg, encoder_channels, decoder_channels, predictor_channels):
    channels_dict={}
    for x in cfg.MODEL.DISCRIMINATOR.TYPE:
        if "Encoder" in x :
            channels_dict[x] = encoder_channels
        elif "Decoder" in x:
            channels_dict[x] = decoder_channels
        elif "Predictor" in x or "Semantic" in x or "Distance" in x:
            if "Pixel" in x:
                if "M" == cfg.MODEL.BACKBONE.TYPE[0]:
                    if "B0" in cfg.MODEL.BACKBONE.TYPE[0]:
                        channels_dict[x] = [256]
                    else:
                        channels_dict[x] = [512]
                else:
                    channels_dict[x] = [2048]
            else:
                channels_dict[x] = predictor_channels
            
    return {x: DISCRIMINATOR[x](cfg=cfg,
                                channels=channels_dict[x],
                                ) for x in cfg.MODEL.DISCRIMINATOR.TYPE}