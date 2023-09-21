import os
import argparse

import torch
from torchvision import utils
import PIL.Image as pilimg
from skimage import img_as_ubyte

from torchvision.utils import make_grid

from model import Generator

import numpy as np

from face_embedding import LResNet


import torchvision.models as models
import torch.nn as nn

def make_video(args):

    # Eigen-Vector
    eigvec = torch.load(args.factor)["eigvec"].to(args.device)

    # =============================================

    # Genearaotr1
    network1 = torch.load(args.ckpt)

    g1 = Generator(256, 512, 8, channel_multiplier=2).to(args.device)
    g1.load_state_dict(network1["g_ema"], strict=False)
    trunc1 = g1.mean_latent(4096)

    # Generator2
    network2 = torch.load(args.ckpt2)

    g2 = Generator(256, 512, 8, channel_multiplier=2).to(args.device)
    g2.load_state_dict(network2["g_ema"], strict=False)
    trunc2 = g2.mean_latent(4096)

    # latent
    if args.seed is not None:
        torch.manual_seed(args.seed)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g1.get_latent(latent)

    # latent direction & scalar
    index=args.index
    degree=args.degree


    # =============================================
    
    images = []

    for deg in range(int(degree)):

        direction = 0.5 * deg * eigvec[:, index].unsqueeze(0)

        img1, _ = g1(
            [latent + direction],
            truncation=args.truncation,
            truncation_latent=trunc1,
            input_is_latent=True,
        )

        img2, _ = g2(
            [latent + direction],
            truncation=args.truncation,
            truncation_latent=trunc2,
            input_is_latent=True,
        )

        grid = make_grid(torch.cat([img1, img2], 0),
                        nrow=args.n_sample,
                        normalize=True,
                        range=(-1,1),
                        )
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = pilimg.fromarray(ndarr)
        images.append(img)

    import imageio
    imageio.mimsave(f'{args.outdir}/{args.video_name}.gif', \
                    [img_as_ubyte(images[i]) \
                    for i in range(len(images))])

def map_func(storage, location):
    return storage.cpu()
    
def where(cond, x_1, x_2):
    return (cond * x_1) + ((~cond) * x_2)
    
def save_image(args):

    # Eigen-Vector
    eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    
    # Genearaotr1
    network1 = torch.load(args.ckpt)

    g1 = Generator(256, 512, 8).to(args.device)
    g1.load_state_dict(network1["g_ema"], strict=False)
    trunc1 = g1.mean_latent(4096).detach().clone()

    network_disney = torch.load('../../../Cartoon-StyleGAN/expr_disney_str/checkpoints/002000.pt')

    g_disney = Generator(256, 512, 8, channel_multiplier=2).to(args.device)
    g_disney.load_state_dict(network_disney["g_ema"], strict=False)
    trunc_disney = g_disney.mean_latent(4096)
    
    network_metface = torch.load('../../../Cartoon-StyleGAN/expr_metface_str/checkpoints/002000.pt')

    g_metface = Generator(256, 512, 8, channel_multiplier=2).to(args.device)
    g_metface.load_state_dict(network_metface["g_ema"], strict=False)
    trunc_metface = g_metface.mean_latent(4096)
    
    network_ukiyoe = torch.load('../../../Cartoon-StyleGAN/expr_ukiyoe/checkpoints/002000.pt')

    g_ukiyoe = Generator(256, 512, 8, channel_multiplier=2).to(args.device)
    g_ukiyoe.load_state_dict(network_ukiyoe["g_ema"], strict=False)
    trunc_ukiyoe = g_ukiyoe.mean_latent(4096)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g1.get_latent(latent)
    
    from parsing import BiSeNet
    parse_model = BiSeNet(n_classes=19)
    parse_model.load_state_dict(torch.load('../../../GAN2Shape/checkpoints/parsing/bisenet.pth', map_location=map_func))
    parse_model = parse_model.cuda()
    parse_model.eval()
    
    FaceEmbedding = LResNet(nch=3, image_size=(128,128), nlayer=18)
    FaceEmbedding = torch.nn.DataParallel(FaceEmbedding).to(args.device)
    FaceEmbedding.load_state_dict(torch.load('../../../LiftedGAN/pretrained/lresnet_casia.pth.tar')['model_state_dict'])
    FaceEmbedding.eval()
    
    g1 = g1.eval()
    
    g_disney = g_disney.eval()
    g_metface = g_metface.eval()
    g_ukiyoe = g_ukiyoe.eval()
    
    num_steps = 10
    lr_rampdown_length = 0.25
    lr_rampup_length = 0.05
    
    # direction
#     direction = (args.degree * eigvec[:, args.index].unsqueeze(0)).clone().detach().float().requires_grad_(True).to(args.device)
    direction1 = args.degree * eigvec[:, args.index].unsqueeze(0)
    direction2 = args.degree * eigvec[:, args.index].unsqueeze(0)
    
    w_opt = torch.tensor(direction1, dtype=torch.float32, device=args.device, requires_grad=True)
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=args.device)
       
    initial_learning_rate = 0.1
    
    direction1.requires_grad = True
    direction2.requires_grad = True
#     latent.requires_grad = True
    optimizer = torch.optim.Adam([direction1] + [direction2], lr=initial_learning_rate)
    
    swap = True 
    swap_layer_num = 2
    
    modules =list(FaceEmbedding.children())[0]
    modules = list(modules.children())[:4]  ## 4-7
    vision_net = nn.Sequential(*modules).cuda().eval()
    
    def logprint(*args):
        print(*args)
    img, save_swap_layer = g1(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc1,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num,
    )
    img1, save_swap_layer1 = g1(
        [latent + direction1],
        truncation=args.truncation,
        truncation_latent=trunc1,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num,
    )
    img2, save_swap_layer2 = g1(
        [latent - direction1],
        truncation=args.truncation,
        truncation_latent=trunc1,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num,
    )
    utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"{args.outdir}/sefa_{args.image_name}_human.png",
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    )
    
    img, _ = g_disney(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc_disney,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
    )
    img1, _ = g_disney(
        [latent + direction1],
        truncation=args.truncation,
        truncation_latent=trunc_disney,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer1,
    )
    img2, _ = g_disney(
        [latent - direction1],
        truncation=args.truncation,
        truncation_latent=trunc_disney,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer2,
    )

    utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"{args.outdir}/sefa_{args.image_name}_disney.png",
        normalize=True,
        range=(-1, 1),
        nrow=1,
    )
    
    img, _ = g_metface(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc_metface,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
    )
    img1, _ = g_metface(
        [latent + direction1],
        truncation=args.truncation,
        truncation_latent=trunc_metface,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer1,
    )
    img2, _ = g_metface(
        [latent - direction1],
        truncation=args.truncation,
        truncation_latent=trunc_metface,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer2,
    )

    utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"{args.outdir}/sefa_{args.image_name}_metface.png",
        normalize=True,
        range=(-1, 1),
        nrow=1,
    )
    
    img, _ = g_ukiyoe(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc_ukiyoe,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
    )
    img1, _ = g_ukiyoe(
        [latent + direction1],
        truncation=args.truncation,
        truncation_latent=trunc_ukiyoe,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer1,
    )
    img2, _ = g_ukiyoe(
        [latent - direction1],
        truncation=args.truncation,
        truncation_latent=trunc_ukiyoe,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer2,
    )

    utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"{args.outdir}/sefa_{args.image_name}_ukiyoe.png",
        normalize=True,
        range=(-1, 1),
        nrow=1,
    )
#     torch.save((latent)[0], '../GAN2Shape/data/Disney_rebuttal/latents/000003.pt')
#     torch.save((latent)[0], '../GAN2Shape/data/Metface_rebuttal/latents/000003.pt')
#     torch.save((latent)[0], '../GAN2Shape/data/ukiyoe_rebuttal/latents/000003.pt')
#     torch.save((latent - direction1)[0], '../GAN2Shape/data/Disney_rebuttal/latents/000004.pt')
#     torch.save((latent - direction1)[0], '../GAN2Shape/data/Metface_rebuttal/latents/000004.pt')
#     torch.save((latent - direction1)[0], '../GAN2Shape/data/ukiyoe_rebuttal/latents/000004.pt')
#     import pdb; pdb.set_trace()
    
    
    index = 0
    
    for step in range(num_steps):
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        img, save_swap_layer = g1(
            [latent],
            truncation=args.truncation,
            truncation_latent=trunc1,
            input_is_latent=True,
            swap=swap, swap_layer_num=swap_layer_num,
        )
        img1, save_swap_layer1 = g1(
            [latent + direction1],
            truncation=args.truncation,
            truncation_latent=trunc1,
            input_is_latent=True,
            swap=swap, swap_layer_num=swap_layer_num,
        )
        img2, save_swap_layer2 = g1(
            [latent - direction1],
            truncation=args.truncation,
            truncation_latent=trunc1,
            input_is_latent=True,
            swap=swap, swap_layer_num=swap_layer_num,
        )

        loss_ID_1 = (FaceEmbedding(img).detach() - FaceEmbedding(img1)).pow(2).sum(1).mean()
        loss_ID_2 = (FaceEmbedding(img).detach() - FaceEmbedding(img2)).pow(2).sum(1).mean()
        loss = (loss_ID_1 + loss_ID_2) / 2

        with torch.no_grad():
            out = parse_model(torch.nn.functional.interpolate(img, 512))[0]
            out = out.argmax(dim=1, keepdim=True)
            mask_all = ((out >= 1) == (out != 16)).float()
            mask_face = ((out >= 1) == (out <= 13)).float()
            mask = (mask_all + mask_face) / 2
            mask = torch.nn.functional.interpolate(mask, 256).repeat(1, 3, 1, 1).bool()

            out1 = parse_model(torch.nn.functional.interpolate(img1, 512))[0]
            out1 = out1.argmax(dim=1, keepdim=True)
            mask_all1 = ((out1 >= 1) == (out1 != 16)).float()
            mask_face1 = ((out1 >= 1) == (out1 <= 13)).float()
            mask1 = (mask_all1 + mask_face1) / 2
            mask1 = torch.nn.functional.interpolate(mask1, 256).repeat(1, 3, 1, 1).bool()
            
            out2 = parse_model(torch.nn.functional.interpolate(img2, 512))[0]
            out2 = out2.argmax(dim=1, keepdim=True)
            mask_all2 = ((out2 >= 1) == (out2 != 16)).float()
            mask_face2 = ((out2 >= 1) == (out2 <= 13)).float()
            mask2 = (mask_all2 + mask_face2) / 2
            mask2 = torch.nn.functional.interpolate(mask2, 256).repeat(1, 3, 1, 1).bool()
        img_mask = where(mask, img, 0)
        img_mask1 = where(mask1, img1, 0)
        img_mask2 = where(mask2, img2, 0)
        img_fea  = vision_net(img_mask)
        img_fea1 = vision_net(img_mask1)
        img_fea2 = vision_net(img_mask2)
        
        pixel_loss_1 = (img_fea.detach() - img_fea1).pow(2).sum(1).mean()
        pixel_loss_2 = (img_fea.detach() - img_fea2).pow(2).sum(1).mean()
        pixel_loss_3 = (img_fea1 - img_fea2).pow(2).sum(1).mean()
        pixel_loss = - torch.log((pixel_loss_1 + pixel_loss_2 + pixel_loss_3) / 3)

        loss = loss + 0.1 * pixel_loss
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: loss {float(loss):<5.2f}, pixel_loss {float(pixel_loss):<5.2f}')
    
        w_out[step] = direction1.detach()[0]

    torch.save((latent + direction1)[index], 'latents/000000.pt')
    torch.save((latent - direction1)[index], 'latents/000001.pt')

    utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"{args.outdir}/ours_{args.image_name}_human_3.png",
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    )
    utils.save_image(
        torch.cat([img1[index].unsqueeze(0), img2[index].unsqueeze(0)], 0),
        f"{args.outdir}/ours_{args.image_name}_human_2.png",
        normalize=True,
        range=(-1, 1),
        nrow=1,
    )
    ###############################################################
    

    img, _ = g_disney(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc_disney,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
    )
    img1, _ = g_disney(
        [latent + direction1],
        truncation=args.truncation,
        truncation_latent=trunc_disney,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer1,
    )
    img2, _ = g_disney(
        [latent - direction1],
        truncation=args.truncation,
        truncation_latent=trunc_disney,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer2,
    )

    utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"{args.outdir}/ours_{args.image_name}_disney.png",
        normalize=True,
        range=(-1, 1),
        nrow=1,
    )
    
    img, _ = g_metface(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc_metface,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
    )
    img1, _ = g_metface(
        [latent + direction1],
        truncation=args.truncation,
        truncation_latent=trunc_metface,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer1,
    )
    img2, _ = g_metface(
        [latent - direction1],
        truncation=args.truncation,
        truncation_latent=trunc_metface,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer2,
    )

    utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"{args.outdir}/ours_{args.image_name}_metface.png",
        normalize=True,
        range=(-1, 1),
        nrow=1,
    )
    
    img, _ = g_ukiyoe(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc_ukiyoe,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
    )
    img1, _ = g_ukiyoe(
        [latent + direction1],
        truncation=args.truncation,
        truncation_latent=trunc_ukiyoe,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer1,
    )
    img2, _ = g_ukiyoe(
        [latent - direction1],
        truncation=args.truncation,
        truncation_latent=trunc_ukiyoe,
        input_is_latent=True,
        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer2,
    )

    utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"{args.outdir}/ours_{args.image_name}_ukiyoe.png",
        normalize=True,
        range=(-1, 1),
        nrow=1,
    )



if __name__ == "__main__":
#     torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=1, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "--factor",
        type=str,
        default="factor.pt",
        help="name of the closed form factorization result factor file",
    )
    parser.add_argument(
        "--save_image",
        action="store_true",
    )
    parser.add_argument(
        "--image_name",
        type=str, default='result',
    )
    # Make GIF
    parser.add_argument(
        "--video",
        action="store_true",
    )
    parser.add_argument(
        "--video_name",
        type=str, default='sefa_video',
    )
    parser.add_argument("--outdir", type=str, default="asset")
    parser.add_argument("--ckpt2", type=str, help="If you make a video, enter the required stylegan2 checkpoints for transfer learning")
    parser.add_argument("--seed", type=int, default=None)
    

    args = parser.parse_args()

    # =============================================

    # directory to save image
    os.makedirs(f'{args.outdir}', exist_ok=True)

    # make video
    if args.save_image == True:
        save_image(args)

    if args.video == True:
        make_video(args)
        