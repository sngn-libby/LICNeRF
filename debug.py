import pytorch_lightning
from pytorch_lightning import Trainer

if __name__ == "__main__":
    t = Trainer(
        log_every_n_steps=1,
        devices=1,
        accelerator="auto",
        replace_sampler_ddp=False,
        strategy=None,
        check_val_every_n_epoch=1,
        precision=16,
    )
    print(t.num_devices)