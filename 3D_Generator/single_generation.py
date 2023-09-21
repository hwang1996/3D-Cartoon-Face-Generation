import os
import argparse

import torch
import torch.nn.functional as F
from torchvision import utils

import sys
sys.path.append('gan2shape/stylegan2/stylegan2-pytorch')
from model import Generator
from tqdm import tqdm


def generate(args, generator1, generator2, device, trunc1, trunc2, swap, swap_layer_num):
    with torch.no_grad():
        generator1.eval()
        generator2.eval()
        count = 0

        f = open(f'{args.save_path}/list.txt', mode='w')

        for count in [0, 1]:
            sample_w = torch.load(f'{args.load_path}/{str(count).zfill(6)}.pt')
            imgs_gen1, save_swap_layer = generator1([sample_w],
                                        input_is_w=True,                                     
                                        truncation=0.7,
                                        truncation_latent=trunc1,
                                        swap=swap, swap_layer_num=swap_layer_num
                                        )
            imgs_gen2, _ = generator2(
                [sample_w],
                truncation=0.7,
                truncation_latent=trunc2,
                input_is_w=True,
                swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer
            )
            utils.save_image(
                    imgs_gen2,
                    f'{args.save_path}/{str(count).zfill(6)}.png',
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
            torch.save(sample_w, f'{args.save_path}/latents/{str(count).zfill(6)}.pt')
            f.write(f'{str(count).zfill(6)}.png \n')

        f.close()


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt1', type=str)
    parser.add_argument('--ckpt2', type=str)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load_path', type=str, default='sample')
    parser.add_argument('--save_path', type=str, default='sample')

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(args.save_path + '/latents')

    # import pdb; pdb.set_trace()

    g_ema1 = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt1)
    g_ema1.load_state_dict(checkpoint['g_ema'], strict=False)

    g_ema2 = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt2)
    g_ema2.load_state_dict(checkpoint['g_ema'], strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent1 = g_ema1.mean_latent(args.truncation_mean)
            mean_latent2 = g_ema2.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    swap = True 
    swap_layer_num = 2

    generate(args, g_ema1, g_ema2, device, mean_latent1, mean_latent2, swap, swap_layer_num)
