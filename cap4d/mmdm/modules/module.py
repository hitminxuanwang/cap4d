from pathlib import Path

import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from controlnet.ldm.util import instantiate_from_config


def tensor_to_img(img):
    img = ((img.permute(1, 2, 0) + 1.) / 2. * 255).clamp(0, 255).detach().cpu().numpy()
    return img[..., [2, 1, 0]].astype(np.uint8)


class CAP4DModule(pl.LightningModule):
    def __init__(
        self, 
        model_config, 
        loss_config,
        # callback_config,
        ckpt_dir,
        *args: pl.Any, 
        **kwargs: pl.Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model = instantiate_from_config(model_config)
        self.loss = instantiate_from_config(loss_config)

        self.ckpt_dir = Path(ckpt_dir)

        self.epoch_id = 0
        
    def training_step(self, batch, batch_idx):
        y = self.model(batch["hint"])

        diff = (batch["jpg"] - y["image"]).abs().mean(dim=1, keepdims=True)
        loss = (diff * y["mask"]).sum() / y["mask"].sum()  # self.loss(batch["jpg"], y)

        return loss
    
    def validation_step(self, batch, batch_idx):
        y = self.model(batch["hint"])

        diff = (batch["jpg"] - y["image"]).abs().mean(dim=1, keepdims=True)
        loss = (diff * y["mask"]).sum() / y["mask"].sum()  # self.loss(batch["jpg"], y)

        if batch_idx == 0:
            epoch_dir = self.ckpt_dir / f"epoch_{self.epoch_id:04d}"
            epoch_dir.mkdir(exist_ok=True)

            print(self.ckpt_dir)
            for i in range(y["image"].shape[0]):
                cv2.imwrite(str(epoch_dir / f"{i:02d}_pred.png"), tensor_to_img(y["image"][i]))
                cv2.imwrite(str(epoch_dir / f"{i:02d}_source.png"), tensor_to_img(batch["hint"]["source_img"][i]))
                cv2.imwrite(str(epoch_dir / f"{i:02d}_target.png"), tensor_to_img(batch["jpg"][i]))

        self.epoch_id += 1

        self.log("loss", loss)
        return loss

    def configure_optimizers(self) -> pl.Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer