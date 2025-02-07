import torch
import torch.nn.functional as F
import torch.nn as nn
from sspengine.engine import BUILDER

class UNetPredictor(nn.Module):
    def __init__(self, encoder_layers, bottle, decoder_layers, out_head, **kargs):
        super().__init__()
        self.encoder_layers = BUILDER.build(encoder_layers)
        self.bottle = BUILDER.build(bottle)
        self.decoder_layers = BUILDER.build(decoder_layers)
        self.out_head = BUILDER.build(out_head)
    
        self.enc_before_hook = kargs['encoder_before_hook'] if 'encoder_before_hook' in kargs.keys() and \
            kargs['encoder_before_hook'] is not None else None
        self.bot_before_hook = kargs['bottle_before_hook'] if 'bottle_before_hook' in kargs.keys() and \
            kargs['bottle_before_hook'] is not None else None
        self.dec_before_hook = kargs['decoder_before_hook'] if 'decoder_before_hook' in kargs.keys() and \
            kargs['decoder_before_hook'] is not None else None
        self.oh_before_hook = kargs['out_head_before_hook'] if 'out_head_before_hook' in kargs.keys() and \
            kargs['out_head_before_hook'] is not None else None
        self.oh_after_hook = kargs['out_head_after_hook'] if 'out_head_after_hook' in kargs.keys() and \
            kargs['out_head_after_hook'] is not None else None

        if isinstance(self.enc_before_hook, dict): self.enc_before_hook = BUILDER.build(self.enc_before_hook)
        if isinstance(self.bot_before_hook, dict): self.bot_before_hook = BUILDER.build(self.bot_before_hook)
        if isinstance(self.dec_before_hook, dict): self.dec_before_hook = BUILDER.build(self.dec_before_hook)
        if isinstance(self.oh_before_hook, dict): self.oh_before_hook = BUILDER.build(self.oh_before_hook)
        if isinstance(self.oh_after_hook, dict): self.oh_after_hook = BUILDER.build(self.oh_after_hook)

    def forward(self, img, **kargs):
        img = self.enc_before_hook(img)
        enc_x, skips = self.encoder_layers(img)
        enc_x, skips = self.bot_before_hook(enc_x, skips)
        bot_x = self.bottle(enc_x)
        bot_x, skips = self.dec_before_hook(bot_x, skips)
        dec_x = self.decoder_layers(bot_x, skips)
        if self.oh_before_hook is not None: dec_x = self.oh_before_hook(dec_x)
        out = self.out_head(dec_x)
        out = self.oh_after_hook(out)
        return out


