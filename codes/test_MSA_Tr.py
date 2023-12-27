import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import models.network_MSA_Tr as swinir
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time

import pyiqa

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize([size,size]))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize([size,size]))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5)),)  
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

  

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,#default='./0006_test.png',
                    help='File path to the content image')
# parser.add_argument('--content_dir', type=str,default='../dataset/LOL/eval15/low/',help='Directory path to a batch of content images')
# parser.add_argument('--content_dir', type=str,default='output/LOL/eval15/low/',help='Directory path to a batch of content images')
parser.add_argument('--content_dir', type=str,default='../dataset/uneven/',help='Directory path to a batch of content images')
# parser.add_argument('--content_dir', type=str,default='../dataset/metergamma/test/low/',help='Directory path to a batch of content images')
# parser.add_argument('--content_dir', type=str,default='../dataset/meterhigh+low/test/low/',help='Directory path to a batch of content images')
# parser.add_argument('--content_dir', type=str,default='../dataset/DICM/',help='Directory path to a batch of content images')
# parser.add_argument('--content_dir', type=str,default='../dataset/LIME/',help='Directory path to a batch of content images')
# parser.add_argument('--content_dir', type=str,default='../dataset/MEF/',help='Directory path to a batch of content images')
# parser.add_argument('--content_dir', type=str,default='../dataset/NPE/',help='Directory path to a batch of content images')
# parser.add_argument('--content_dir', type=str,default='../dataset/VV/',help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,#default='./0006_test.png',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,default='../dataset/LOL/eval15/high/',help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='output/',
                    help='Directory to save the output image(s)')
parser.add_argument('--vgg', type=str, default='./experiments/nocross256/vgg_normalised.pth')
parser.add_argument('--gen_path', type=str, default='experiments/Gen_iter_last.pth')


parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()




# Advanced options
content_size=512
style_size=512
crop='store_true'
save_ext='.jpg'
output_path=args.output
output_path = output_path+args.content_dir[11:]
preserve_color='store_true'
alpha=args.a

s_psnr = 0
s_ssim = 0
s_niqe = 0
s_lpips = 0
niqe_metric = pyiqa.create_metric('niqe', device=torch.device('cuda'))
lpips_metric = pyiqa.create_metric('lpips', device=torch.device('cuda'))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
iqa_ssim = pyiqa.create_metric('ssim', device=device)

if args.content_dir != '../dataset/LOL/eval15/low/':
    args.style_dir = args.content_dir
# Either --content or --content_dir should be given.
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --style_dir should be given.
if args.style:
    style_paths = [Path(args.style)]    
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

# if not os.path.exists(output_path):
#     os.mkdir(output_path)


window_size = 2
height = 256
width = 256
with torch.no_grad():
    #Gen
    # network = swinir.SwinIR(upscale=1, img_size=(height, width),
    #                window_size=window_size, img_range=1., depths=[2,2,2],
    #                embed_dim=30, num_heads=[3,3,3], mlp_ratio=2, upsampler='pixelshuffledirect')
    network = swinir.SwinIR(img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[1,1,1],
                   embed_dim=32, num_heads=[4, 4,4], mlp_ratio=2)
    # network = swinir.SwinIR(upscale=1, img_size=(height, width),
    #                window_size=window_size, img_range=1., depths=[6,6,6,6],
    #                embed_dim=60, num_heads=[6, 6,6,6], mlp_ratio=2, upsampler='pixelshuffledirect')
network.eval()
from collections import OrderedDict
new_state_dict = OrderedDict()
state_dict = torch.load(args.gen_path)
for k, v in state_dict.items():
    namekey = k[7:] # remove `module.`
    # namekey = k
    new_state_dict[namekey] = v
network.load_state_dict(new_state_dict)
network.to(device)
network = nn.DataParallel(network, device_ids=[0])



content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)
re_tf = style_transform(content_size,crop)

i = 0
for content_path,style_path in zip(content_paths,style_paths):
    i+=1
    print(content_path,style_path)
  
    content = content_tf(Image.open(content_path).convert("RGB"))

    h,w,c=np.shape(content)    
    style = style_tf(Image.open(style_path).convert("RGB"))

      
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    
    start = time.time()

    with torch.no_grad():
        output= network(content)  

    end_time = (time.time() - start)
    print(end_time)

    #NIQE and lpips
    niqe = niqe_metric(output)
    lpips = lpips_metric(output,style)
    s_niqe += niqe
    s_lpips += lpips

    #pnsr and ssim
    fake_high = output.permute(0, 2, 3, 1).data.cpu().numpy()
    high = style.permute(0, 2, 3, 1).data.cpu().numpy()
    fake_high = np.minimum(np.maximum(fake_high, 0.0), 1.0)
    high = np.minimum(np.maximum(high, 0.0), 1.0)
    temp_out = np.uint8(fake_high[0] * 255)
    temp_high = np.uint8(high[0] * 255)
    psnr = compare_psnr(np.array(temp_out), np.array(temp_high))
    # psnr = iqa_metric(output, style)
    ssim = compare_ssim(np.array(temp_out), np.array(temp_high), multichannel=True)
    # ssim = iqa_ssim(output, style)
    s_psnr += psnr
    s_ssim += ssim

    output = output[0]
    # output = content
    # output = style
    # output = torch.cat((content,output),0)
    output = output.cpu()        
    # output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
    #     output_path, splitext(basename(content_path))[0],
    #     splitext(basename(style_path))[0], save_ext
    #     )
    output_name = str(content_path).replace('../dataset/','output/')
    if not os.path.exists(output_name.replace('/'+output_name.split("/")[-1],'')):
        os.makedirs(output_name.replace('/'+output_name.split("/")[-1],''))
    # sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
    save_image(output, output_name)
    # print("%d/%d, psnr=%.4f, ssim=%.4f, lpips=%.4f, NIQE=%.4f" % (i, len(dataloader), psnr, ssim, lpips, niqe))
    print("%d/%d, psnr=%.4f, ssim=%.4f, lpips=%.4f, NIQE=%.4f" % (i, len(content_paths), psnr, ssim, lpips, niqe))
print("Test, psnr=%.4f, ssim=%.4f, lpips=%.4f, NIQE=%.4f" % (s_psnr / len(content_paths), s_ssim / len(content_paths), s_lpips / len(content_paths), s_niqe / len(content_paths)))
   

