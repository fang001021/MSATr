import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import models.network_MSA_Tr as swin_model
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
from torch.autograd import Variable
import function
from function import pic_mix
import pytorch_ssim
from torchvision.models.vgg import vgg16

def train_transform():
    transform_list = [
        # transforms.Resize(size=(256, 256)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        # transforms.Normalize((0), (1)),
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)             
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    # lr = args.lr *0.1* 4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_dislearning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr *0.1 / (1.0 + args.lr_decay * iteration_count)
    # lr = args.lr  / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def ssim_loss(input, gt):
    sm_loss = pytorch_ssim.ssim(input, gt)
    return 1 - sm_loss

Ltv_loss = function.L_TV()

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='../dataset/LOL/our485/low', type=str,   
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='../dataset/LOL/our485/high', type=str,  #wikiart dataset crawled from https://www.wikiart.org/
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./experiments1/vgg_normalised.pth')  #run the train.py, please download the pretrained vgg checkpoint

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=800000)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--cuda', default=True, help='use GPU computation')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

vgg = vgg16(pretrained=True)
loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
loss_network = loss_network.cuda()
for param in loss_network.parameters():
    param.requires_grad = False
mse_loss = nn.MSELoss()

def vgg_loss(input, gt):
    perception_loss = mse_loss(loss_network(input), loss_network(gt))
    return perception_loss

upscale = 1
window_size = 2
height = 256
width = 256

with torch.no_grad():
    #Gen
    network = swin_model.SwinIR(img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[1,1,1],
                   embed_dim=32, num_heads=[4,4,4], mlp_ratio=2)
    #Dis
    Dis = swin_model.Discriminator(3)
    Dis_local = swin_model.Discriminator(3)

network.train()
network.to(device)
network = nn.DataParallel(network, device_ids=[0])

Dis.train()
Dis.to(device)
Dis = nn.DataParallel(Dis, device_ids=[0])

Dis_local.train()
Dis_local.to(device)
Dis_local = nn.DataParallel(Dis_local, device_ids=[0])

content_tf = train_transform()
style_tf = train_transform()





content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
 

optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
optimizerdis = torch.optim.Adam(Dis.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizerdis_local = torch.optim.Adam(Dis_local.parameters(), lr=args.lr, betas=(0.5, 0.999))

if not os.path.exists(args.save_dir+"/test"):
    os.makedirs(args.save_dir+"/test")

Tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor
target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)
target_real_L = Variable(Tensor(args.batch_size*4).fill_(1.0), requires_grad=False)
target_fake_L = Variable(Tensor(args.batch_size*4).fill_(0.0), requires_grad=False)

for i in tqdm(range(args.max_iter)):

    # if i < 1e4:
    #     warmup_learning_rate(optimizer, iteration_count=i)
    # else:
    #     adjust_learning_rate(optimizer, iteration_count=i)

    adjust_learning_rate(optimizer, iteration_count=i)
    adjust_dislearning_rate(optimizerdis, iteration_count=i)
    adjust_dislearning_rate(optimizerdis_local, iteration_count=i)

    # print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)  
    content_high = network(content_images)
    style_low = network(style_images)

    content_local = function.split_Local(content_high,args.size,32)
    style_local = function.split_Local(style_images,args.size,32)

    #GAN loss
    criterion_GAN = torch.nn.MSELoss()
    pred = Dis(content_high)
    pred_Local = Dis_local(content_local)
    loss_s = criterion_GAN(pred.squeeze(), target_real)+criterion_GAN(pred_Local.squeeze(), target_real_L)
    # loss_s = loss_s*2.0

    # uneven loss
    ue_low,ue_high,uesize = pic_mix(content_images,content_high)
    ue_img_high = network(ue_high)
    ue_img_low = network(ue_low)
    # loss_uneven = L_TV(ue_img_high,content_high)/uesize+L_TV(ue_img_low,content_high)/uesize
    loss_uneven = mse_loss(ue_img_high,content_high)/uesize+mse_loss(ue_img_low,content_high)/uesize


    #identity loss
    loss_identity_style  = ssim_loss(style_images,style_low)

    #content loss
    loss_c = vgg_loss(content_images,content_high)

    loss_ex = function.calc_expose_loss(content_high)
    if i<=15000:
        loss = loss_identity_style+loss_s +loss_c        
    else:
        loss = loss_identity_style+loss_s +loss_c+loss_uneven#+loss_ex
    # loss = loss_s +loss_c+loss_ex+loss_uneven
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    if i % 200 == 0:
        output_name = '{:s}/test/{:s}{:s}'.format(
                        args.save_dir, str(i),".jpg"
                    )
        out1 = torch.cat((content_images,style_images),0)
        out2 = torch.cat((content_high,style_low),0)
        out3 = torch.cat((ue_low,ue_img_low),0)
        out4 = torch.cat((ue_high,ue_img_high),0)
        out = torch.cat((out1,out2,out3,out4),2)
        save_image(out, output_name)

    ###### Discriminator  ######
    optimizerdis.zero_grad()

    # Real loss
    pred_real = Dis(style_images)
    loss_D_real = criterion_GAN(pred_real.squeeze(), target_real)

    # Fake loss
    pred_fake = Dis(content_high.detach())
    loss_D_fake = criterion_GAN(pred_fake.squeeze(), target_fake)

    # Total loss
    loss_D_A = (loss_D_real + loss_D_fake)
    loss_D_A.backward()

    optimizerdis.step()
    ###################################
    loss_GAN = loss_D_A

    ###### Discriminator local ######
    optimizerdis_local.zero_grad()

    # Real loss
    pred_real = Dis_local(style_local)
    loss_D_real = criterion_GAN(pred_real.squeeze(), target_real_L)

    # Fake loss
    pred_fake = Dis_local(content_local.detach())
    loss_D_fake = criterion_GAN(pred_fake.squeeze(), target_fake_L)

    # Total loss
    loss_D_L = (loss_D_real + loss_D_fake)
    loss_D_L.backward()

    optimizerdis_local.step()
    ################################### 

    print(loss.sum().cpu().detach().numpy(),
            "-HH:",loss_identity_style.sum().cpu().detach().numpy(),
            "-content:",loss_c.sum().cpu().detach().numpy(),
            "-uneven:",loss_uneven.sum().cpu().detach().numpy(),
            "-GAN:",loss_s.sum().cpu().detach().numpy(),
            "-expose:",loss_ex.sum().cpu().detach().numpy(),
            "-Dis:",loss_GAN.sum().cpu().detach().numpy(),
            "-loss_D_L:",loss_D_L.sum().cpu().detach().numpy(),
              )
    
    writer.add_scalar('loss_HH', loss_identity_style.sum().item(), i + 1)
    writer.add_scalar('total_loss', loss.sum().item(), i + 1)
    writer.add_scalar('GAN_loss', loss_s.sum().item(), i + 1)
    writer.add_scalar('Dis_loss', loss_GAN.sum().item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        torch.save(network.state_dict(),
                   '{:s}/Gen_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        torch.save(Dis.state_dict(),'{:s}/Dis_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        torch.save(Dis_local.state_dict(),'{:s}/Dis_local_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        torch.save(network.state_dict(),
                   '{:s}/Gen_iter_last.pth'.format(args.save_dir))                                   
writer.close()


