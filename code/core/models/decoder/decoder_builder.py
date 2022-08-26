from ..registry import DECODER

def build_decoder(cfg, input_channels):
    assert cfg.MODEL.DECODER.TYPE in DECODER, \
        "cfg.MODEL.DECODER.TYPE: {} are not registered in registry".format(
            cfg.MODEL.DECODER.TYPE
        )
    if "DeepLab" in cfg.MODEL.DECODER.TYPE:
        return DECODER[cfg.MODEL.DECODER.TYPE](input_channels, cfg.MODEL.PREDICTOR.NUM_CLASSES)
    elif "SegFormer" in cfg.MODEL.DECODER.TYPE:
        if "B0" in cfg.MODEL.BACKBONE.TYPE:
            in_channels = [32, 64, 160, 256]
            embed_dim = 256
        elif "B1" in cfg.MODEL.BACKBONE.TYPE:
            in_channels = [64, 128, 320, 512]
            embed_dim = 256
        else:
            in_channels = [64, 128, 320, 512]
            embed_dim = 768
        feature_strides=[4, 8, 16, 32]
        
        return DECODER[cfg.MODEL.DECODER.TYPE](in_channels, feature_strides, embed_dim, cfg.MODEL.PREDICTOR.NUM_CLASSES)
    
    return DECODER[cfg.MODEL.DECODER.TYPE](input_channels)