import os
from pprint import pprint
from configs.config_ht_vt import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks

from models.UCARE import UCARE



from lightning.pytorch import seed_everything
import lightning.pytorch as pl
import torch

# os.environ["XFORMERS_USE"] = "0"  # 禁用xFormers
torch.backends.cudnn.deterministic = True

# export CUDA_VISIBLE_DEVICES=1
def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
     
        devices=[1],
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"],
        gradient_clip_val=1.0,  # 设置梯度裁剪的最大范数 
        logger=callbacks["loggers"]
    )
    
    # model = R2GenGPT()

    if args.ckpt_file is not None:
        # model = R2GenGPT.load_from_checkpoint(args.ckpt_file, strict=False)
        model = R2GenGPT(args)
         # state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:1'))['state_dict']
            # self.load_state_dict(state_dict=state_dict, strict=False)
        state_dict = torch.load(args.ckpt_file, map_location="cpu",weights_only=False)['state_dict']
        msg = model.load_state_dict(state_dict, strict=False)
        print("load checkpoint from {}".format(args.ckpt_file))
    else:
        model = R2GenGPT(args)
    


    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:   
        trainer.fit(model, datamodule=dm)
        
        trainer.test(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)

    train(args)


if __name__ == '__main__':
    main()