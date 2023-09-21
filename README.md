# [3D Cartoon Face Generation with Controllable Expressions from a Single GAN Image](https://arxiv.org/abs/2207.14425)
## Codes of ACME (PyTorch)
*3D Cartoon Face Generation with Controllable Expressions from a Single GAN Image*  
Hao Wang, Guosheng Lin, Steven C. H. Hoi, Chunyan Miao  


<!-- ![outline](https://github.com/hwang1996/ACME/blob/master/imgs/cvpr_fig.png) -->

If you find this code useful, please consider citing:
```
@article{wang20223d,
  title={3D Cartoon Face Generation with Controllable Expressions from a Single GAN Image},
  author={Wang, Hao and Lin, Guosheng and Hoi, Steven CH and Miao, Chunyan},
  journal={arXiv preprint arXiv:2207.14425},
  year={2022}
}
```


## 2D Cartoon Generation Model Training
```
cd Cartoon_Generator
python train.py --batch=2 --ckpt=ffhq256.pt --structure_loss=1 --freezeD=2 --augment --path=data/disney_lmdb/
```

## Facial Expression Manipulation
```
cd Cartoon_Generator
python apply_factor_opt.py --ckpt=ffhq256.pt --save_image
```

## 3D Shape Generation

### Data Preparation
```
cd 3D_Generator
python single_generation.py --ckpt1 ffhq256.pt --ckpt2 disney.pt --load_path ../Cartoon_Generator/latents/ --save_path data
```
### Training
```
bash scripts/run_sample.sh
```

## Pretrained Model
We provide the [pretrained models](https://hkustgz-my.sharepoint.com/:f:/g/personal/haowang_hkust-gz_edu_cn/EiE2yIi1729OrvmVAvr7xx0BRZzguygZ_bxqnovR_q1weA?e=pOqraq). 

## Acknowledgement
Part of the code is borrowed from [GAN2Shape](https://github.com/XingangPan/GAN2Shape) and [Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN). Please the installation requirements from these repositories. 
