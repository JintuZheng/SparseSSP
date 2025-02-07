from sspengine.engine import BUILDER
from sspengine.models.unet_predictor import UNetPredictor
import torch

class TaskUNetPredictor(UNetPredictor):
    def __init__(self, task_emb_func, encoder_layers, bottle, decoder_layers, out_head, **kargs):
        super().__init__(encoder_layers, bottle, decoder_layers, out_head, **kargs)
        self.task_emb_func = task_emb_func

    def forward(self, img, task_id, **kargs):
        task_emb = self.task_emb_func(task_id, self.encoder_layers.num_tasks)
        img = self.enc_before_hook(img, task_emb)
        enc_x, skips = self.encoder_layers(img, task_emb)
        enc_x = self.bot_before_hook(enc_x, task_emb)
        bot_x = self.bottle(enc_x, task_emb)
        bot_x, skips = self.dec_before_hook(bot_x, skips, task_emb)
        dec_x = self.decoder_layers(bot_x, skips, task_emb)
        dec_x = self.oh_before_hook(dec_x)
        out = self.out_head(dec_x, task_emb)
        out = self.oh_after_hook(out)
        return out


