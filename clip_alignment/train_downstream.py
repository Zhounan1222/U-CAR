import os
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.swin_clip_DownStream import swin_CLIP_DownStream
from lightning.pytorch import seed_everything
import lightning.pytorch as pl
# from models.swin_clip_DownStream_add_label import swin_CLIP_DownStream_label
# from models.swin_clip_DownStream_add_visual import swin_CLIP_DownStream_visual
# from models.swin_clip_DownStream_add_similar_report import swin_CLIP_DownStream_report

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
        logger=callbacks["loggers"]
    )

    model = swin_CLIP_DownStream(args)
    # model = swin_CLIP_DownStream_label(args)
    # model = swin_CLIP_DownStream_visual(args)
    # model =swin_CLIP_DownStream_report(args)
    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)


if __name__ == '__main__':
    main()