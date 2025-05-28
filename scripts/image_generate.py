"""
Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""

import argparse

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
import os

import itertools

from DCGG.image_datasets import MaskGenerator, load_conditions
from DCGG import dist_util, logger
from DCGG.ssim import ssim_color, Edge_select_faiss, histogram_similarity, apply_mask
from DCGG.get_vgg_model import get_vgg_model
from DCGG.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def exclude_bias_and_norm(p):
    return p.ndim == 1

def main(args):
    base_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(base_dir)
    vgg_medel_path= os.path.join(parent_dir, "model/vgg_normalised.pth") 
    args.out_dir = parent_dir+'/result/sample'
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Load model
    if args.model_name == "Chimanimani_Zimbabwe":
        model_path=parent_dir+'/model/Chimanimani_Zimbabwe.pt'
        A_dir=parent_dir+'/Dataset/Chimanimani_Zimbabwe/A'
        B_dir=parent_dir+'/Dataset/Chimanimani_Zimbabwe/B'
    elif args.model_name == "Kurucasile_Turkey":
        model_path=parent_dir+'/model/Kurucasile_Turkey.pt'
        A_dir=parent_dir+'/Dataset/Kurucasile_Turkey/A'
        B_dir=parent_dir+'/Dataset/Kurucasile_Turkey/B'
    else:
        raise ValueError("unrecognizable model name")

    args.gpu = 1
    th.cuda.set_device(1)
    logger.configure(dir=args.out_dir)
    logger.log("Load data...")
    data = load_conditions(
        A_dir=A_dir,
        B_dir=B_dir,
        mask_dir=args.label_dir,
        image_size=args.image_size,
    )

    # Use features conditioning
    
    vgg_model = get_vgg_model(vgg_medel_path).cuda().eval()
    
    for p in vgg_model.parameters():
        vgg_model.requires_grad = False # type: ignore

    # ============ preparing data ... ============
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    

    trained_model = th.load(model_path, map_location="cpu")
    model.load_state_dict(trained_model, strict=True)
    model.to(dist_util.dev())
    model.eval()

    # Choose first image
    logger.log("sampling...")
    all_images = []
    all_A = []
    all_masks = []
    all_B = []
    path_B= args.out_dir+'/B'
    if not os.path.exists(path_B):
        os.makedirs(path_B)
    path_A= args.out_dir+'/A'
    if not os.path.exists(path_A):
        os.makedirs(path_A)
    path_label= args.out_dir+'/label'
    if not os.path.exists(path_label):
        os.makedirs(path_label)

    sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
    num_current_samples = 0
    data_iter = itertools.cycle(data)

    while num_current_samples < args.num_images:

        if args.label_dir == "":
            imageA_batch, imageB_batch = next(data_iter)
            mask_batch = MaskGenerator(image_size=args.image_size, batch_size=args.batch_size)
        else:
            imageA_batch, imageB_batch, maskmap = next(data_iter)
            mask_batch = maskmap[0:1].repeat(args.batch_size, 1, 1, 1)
        batch_A = imageA_batch[0:1].repeat(args.batch_size, 1, 1, 1).cuda()  
        batch_B = imageB_batch[0:1].repeat(args.batch_size, 1, 1, 1).cuda()  
        model_kwargs = {}

        with th.no_grad():
            model_kwargs["context"] = vgg_model(batch_A.to(args.gpu))

        
        bs, _, h, w = mask_batch.size()
        input_label = th.zeros((bs, 2, h, w), dtype=th.float32)
        input_semantics = input_label.scatter_(1, mask_batch, 1.0)
        input_semantics = input_semantics.cuda()
        mask_batch = mask_batch.cuda()
        model_kwargs["mask"] = input_semantics

        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        if args.use_ddim:
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        else:
            sample = ((sample[-1] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        samples = sample.contiguous()
        all_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in samples])

        A_image = ((batch_A[0:1] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        A_image = A_image.permute(0, 2, 3, 1)
        A_images = A_image.contiguous()
        all_A.extend([sample.unsqueeze(0).cpu().numpy() for sample in A_images])
        

        B_image = (batch_B[0:1]).clamp(0, 255).to(th.uint8)
        B_image = B_image.permute(0, 2, 3, 1)
        B_images = B_image.contiguous()
        all_B.extend([sample.unsqueeze(0).cpu().numpy() for sample in B_images])

        mask_batch = (mask_batch* 255.0).clamp(0, 255).to(th.uint8)
        mask_batch = mask_batch.permute(0, 2, 3, 1)
        mask_batch = mask_batch.contiguous()
        all_masks.extend([masks.unsqueeze(0).cpu().numpy() for masks in mask_batch])

        logger.log(f"created {num_current_samples+1} samples")
        

        num_current_samples += 1

    for n in range(0,args.num_images):
        score_best=0
        index=0
        img_B=all_B[n]
        for i in range(0,args.batch_size):


            img_data = all_images[n*args.batch_size+i]
            mask_data = all_masks[n*args.batch_size+i]
            img_data = np.squeeze(img_data)
            mask_data = np.squeeze(mask_data)
            img_B = np.squeeze(img_B)
            edge_score = Edge_select_faiss(img_data,mask_data)
            mask_data = mask_data.astype(np.float32) / 255.0

            masked_gen = apply_mask(img_data, mask_data)
            masked_target = apply_mask(img_B, mask_data)

            ssim_score = ssim_color(masked_target, masked_gen)
            hist_score = histogram_similarity(masked_target, masked_gen)
            score = 0.7 * (0.5*ssim_score+0.5*edge_score) + 0.3 * hist_score

            if score>score_best:
                score_best=score
                index=i

        ######################################################
        img_data=all_images[n*args.batch_size+index]
        mask_data=all_masks[n*args.batch_size+index]
        img_A=all_A[n]
    
        img_data = np.squeeze(img_data, axis=0)
        img = Image.fromarray(img_data).convert('RGB')
        img_A = np.squeeze(img_A, axis=0)
        A = Image.fromarray(img_A).convert('RGB')
        mask_data = np.squeeze(mask_data)
        mask = Image.fromarray(mask_data).convert('L')
        img.save(path_B+f'/{n}.png')
        A.save(path_A+f'/{n}.png')
        mask.save(path_label+f'/{n}.png')
    


    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_images=600,
        batch_size=12,
        use_ddim=True,
        submitit=False,
        local_rank=0,
        dist_url="env://",
        G_shared=False,
        timestep_respacing = 80,
        
    )
    defaults.update(model_and_diffusion_defaults()) # type: ignore
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default="", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--label_dir', default="", type=str, help='')
    parser.add_argument('--model_name', default="Chimanimani_Zimbabwe", type=str, help='')
    parser.add_argument('--no_shared', action='store_false', default=True,
                        help='This flag enables squeeze and excitation.')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
