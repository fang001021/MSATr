import torch
import torch.nn as nn
import random
import torch.nn.functional as F

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_mean_std1(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    # assert (len(size) == 4)
    WH,N, C = size
    feat_var = feat.var(dim=0) + eps
    feat_std = feat_var.sqrt()
    feat_mean = feat.mean(dim=0)
    return feat_mean, feat_std
def normal(feat, eps=1e-5):
    feat_mean, feat_std= calc_mean_std(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized 
def normal_style(feat, eps=1e-5):
    feat_mean, feat_std= calc_mean_std1(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

class Expose_loss(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(Expose_loss, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d
    
def split_Local(pic,size0,split_size):
    Local_B = torch.split(pic,split_size,dim=2)
    Local_B = torch.cat((Local_B[random.randint(0,size0/split_size-1)],Local_B[random.randint(0,size0/split_size-1)]),dim = 0)
    Local_B = torch.split(Local_B,split_size,dim=3)
    Local_B = torch.cat((Local_B[random.randint(0,size0/split_size-1)],Local_B[random.randint(0,size0/split_size-1)]),dim = 0)
    return Local_B

def pic_mix(low, high):
    n,c,_,size = low.size()
    #窗口的大小
    w_h = random.randint(size/4,3*size/4)
    w_w = random.randint(size/4,3*size/4)
    #窗口蒙版的左上位置
    h_st = random.randint(0,size-w_h-1)
    w_st = random.randint(0,size-w_w-1)
    x = random.random()
    i = random.randint(0,1)
    #mix
    pic_low = low.clone()
    # pic[:,:,h_st:h_st+w_h,w_st:w_st+w_w] = high[:,:,h_st:h_st+w_h,w_st:w_st+w_w]
    pic_low[:,:,h_st:h_st+w_h,w_st:w_st+w_w] = x*high[:,:,h_st:h_st+w_h,w_st:w_st+w_w]+(1-x)*low[:,:,h_st:h_st+w_h,w_st:w_st+w_w]
    pic_high = high.clone()
    # pic[:,:,h_st:h_st+w_h,w_st:w_st+w_w] = high[:,:,h_st:h_st+w_h,w_st:w_st+w_w]
    pic_high[:,:,h_st:h_st+w_h,w_st:w_st+w_w] = (1-x)*high[:,:,h_st:h_st+w_h,w_st:w_st+w_w]+x*low[:,:,h_st:h_st+w_h,w_st:w_st+w_w]
    return pic_low,pic_high,x#w_h*w_w/size/size*x*x

#曝光照明损失
def calc_expose_loss(input):
        expose_loss = Expose_loss(16,0.6)
        return(expose_loss(input))

#照明平滑度损失
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    