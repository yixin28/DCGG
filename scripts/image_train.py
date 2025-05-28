"""
Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
Train a conditional (representation based) diffusion model on images.
"""
import os
import argparse
import torch as th
from DCGG import dist_util, logger
from DCGG.image_datasets import load_data
from DCGG.resample import create_named_schedule_sampler
from DCGG.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from DCGG.train_util import TrainLoop
from DCGG.get_vgg_model import get_vgg_model
from torch.cuda.amp import autocast

def main(args):
    base_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(base_dir)
    vgg_medel_path= os.path.join(parent_dir, "model/vgg_normalised.pth") 
    args.out_dir = parent_dir+'/result/train'
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    dist_util.init_distributed_mode(args)

    logger.configure(dir=args.out_dir)
    
    vgg_model = get_vgg_model(vgg_medel_path).to(args.gpu).eval()
    for _,p in vgg_model.named_parameters():
        p.requires_grad_(False)

    # Create model
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))

    if args.model_path != "":
        trained_model = th.load(args.model_path, map_location="cpu")
        model.load_state_dict(trained_model, strict=True)
    model.to(args.gpu)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Create the dataloader
    logger.log("creating data loader...")
    data = load_all_data(
        args,
        vgg_model=vgg_model,
    )

    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        gpu = args.gpu,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps
    ).run_loop()

def load_all_data(args, vgg_model=None):
    data = load_data(
        A_dir=args.A_dir,
        mask_dir=args.label_dir,
        B_dir=args.B_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    for batch, batch_A, model_kwargs in data:
        if vgg_model is not None:
            with th.no_grad():
                with autocast(args.use_fp16):
                    model_kwargs["context"] = vgg_model(batch_A.to(args.gpu))
            yield batch, model_kwargs
        else:
            yield batch, model_kwargs

def create_argparser():
    
    defaults = dict(
        A_dir="",
        B_dir="",
        label_dir="",
        model_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=3000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        submitit=False,
        local_rank=0,
        dist_url="env://",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out_dir', default="", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--no_shared', action='store_false', default=True,
                        help='Disable the shared lower dimensional projection of the representation.')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
