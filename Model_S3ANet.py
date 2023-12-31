import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange


class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()        
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

def adjust_learning_rate(optimizer,base_lr, i_iter, max_iter, power=0.9):
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def CC_module(proj_query, proj_key, proj_value):

    m_batchsize, _, height, width = proj_value.size()

    proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous(). \
        view(m_batchsize * width, -1, height).permute(0, 2, 1)
    proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous(). \
        view(m_batchsize * height, -1, width).permute(0, 2, 1)

    proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
    proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
    proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
    proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

    A1 = proj_query_H / (torch.sqrt(torch.sum(torch.mul(proj_query_H, proj_query_H), dim=-1, keepdim=True))+1e-10)
    B1 = proj_key_H / (torch.sqrt(torch.sum(torch.mul(proj_key_H, proj_key_H), dim=1, keepdim=True))+1e-10)
    energy_H = torch.bmm(A1, B1).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)

    A2 = proj_query_W / (torch.sqrt(torch.sum(torch.mul(proj_query_W, proj_query_W), dim=-1, keepdim=True))+1e-10)
    B2 = proj_key_W / (torch.sqrt(torch.sum(torch.mul(proj_key_W, proj_key_W), dim=1, keepdim=True))+1e-10)
    energy_W = torch.bmm(A2, B2).view(m_batchsize, height, width, width)
    concate = F.softmax(torch.cat([energy_H, energy_W], 3), 3)

    att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
    att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
    out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
    out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
    return out_H + out_W

def grid_PA(cont_p, query, key, value, bin, START_H, END_H, START_W, END_W):

    for i in range(0, bin):
        for j in range(0, bin):
            cont_p=cont_p.clone()
            value = value.clone()
            key = key.clone()
            query = query.clone()

            value_local = value[:, :, START_H[i, j]:END_H[i, j], START_W[i, j]:END_W[i, j]]

            #print(i,j, START_H[i, j],END_H[i, j], START_W[i, j], END_W[i, j])
            #value_local.backward(torch.ones(1, 64, 256 // bin, 108 // bin).cuda(), retain_graph=True) # backword error checking

            query_local = query[:, :, START_H[i, j]:END_H[i, j], START_W[i, j]:END_W[i, j]]
            key_local = key[:, :, START_H[i, j]:END_H[i, j], START_W[i, j]:END_W[i, j]]

            cont_p_local = CC_module(query_local, key_local, value_local)

            cont_p[:, :, START_H[i, j]:END_H[i, j], START_W[i, j]:END_W[i, j]] = cont_p_local

    return cont_p

class GST_block(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                GST(dim=dim, num_heads=heads,bias=False),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        # x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x.permute(0, 2, 3, 1)) + x.permute(0, 2, 3, 1)
        out = x.permute(0, 3, 1, 2)
        # out = x.permute(0, 3, 1, 2)
        return out

class GST(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GST, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class PPM_Spa(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM_Spa, self).__init__()
        # num = len(bins)
        # self.args = args

        self.conv1=nn.Conv2d(in_dim,reduction_dim,1, bias=False)
        self.fuc_pc = nn.ModuleList()
        for bin in bins:
            self.fuc_pc.append(FuCont_PSP_Spa(reduction_dim, bin))


        self.conv = nn.Conv2d(in_dim+256, reduction_dim, 1, bias=False)
        self.gn = nn.GroupNorm(16,reduction_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        out = [x]
        L = []
        x = self.conv1(x)  # dim reduction
        for path in self.fuc_pc:
            L.append(path(x))
        out.extend(L)

        out = torch.cat(out, 1)
        out = self.conv(out)
        out = self.gn(out)
        out = self.relu(out)
        return out

class FuCont_PSP_Spa(nn.Module):
    def __init__(self, in_dim, bin):
        super(FuCont_PSP_Spa, self).__init__()
        # self.args = args
        self.bin = bin
        self.query_conv_p = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv_p = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv_p = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

    def forward(self, x):

        _, _, h, w = x.size()

        step_h, step_w = h // self.bin, w // self.bin

        START_H = np.zeros([self.bin, self.bin]).astype(int)
        END_H = np.zeros([self.bin, self.bin]).astype(int)
        START_W = np.zeros([self.bin, self.bin]).astype(int)
        END_W = np.zeros([self.bin, self.bin]).astype(int)

        for i in range(0, self.bin):
            for j in range(0, self.bin):
                start_h, start_w = i * step_h, j * step_w
                end_h, end_w = min(start_h + step_h, h), min(start_w + step_w, w)
                if i == (self.bin - 1):
                    end_h = h
                if j == (self.bin - 1):
                    end_w = w
                START_H[i, j] = start_h
                END_H[i, j] = end_h
                START_W[i, j] = start_w
                END_W[i, j] = end_w

        cont_p = torch.zeros(x.shape).cuda()

        for cnt in range(2):
            value = self.value_conv_p(x)
            query = self.query_conv_p(x)
            key = self.key_conv_p(x)

            cont_p = grid_PA(cont_p, query, key, value, self.bin, START_H, END_H, START_W, END_W)

            x = cont_p  # recurrent
        return x

class S3ANet(nn.Module):
    def __init__(self, num_features=103, num_classes=9, conv_features=64,
                 bins=[1,2,3,6],in_dim=64,image_size=13,dim=1024):
        super(S3ANet, self).__init__()

        self.conv0 = nn.Conv2d(num_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=1,
                               bias=True)
        self.conv1 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=2,
                               bias=True)
        self.conv2 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=3,  # 3
                               bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.dim = dim

        self.conv_cls = nn.Conv2d(conv_features * 3, num_classes, kernel_size=1, stride=1, padding=0,
                                  bias=True)
        self.drop = nn.Dropout(0.5)
        self.conv_features = conv_features
        self.num_features = num_features
        self.bin = bin
        self.in_dim = in_dim
        self.image_size = image_size
        self.GST_block = GST_block(dim=64,heads=16,num_blocks=1)
        self.head = PPM_Spa(conv_features,conv_features,bins)

    def re_init(self):
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim)).cuda()

    def forward(self, x, mask=None):
        ################## backbone ##############
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        x_copy = x.squeeze()
        x = self.relu(self.conv0(x))
        conv1 = x

        x = self.relu(self.conv1(x))
        conv2 = x
        x = self.avgpool(x)

        x = self.relu(self.conv2(x))
        x1 = x # skip connection
        n, c, h, w = x.size()  # m* h/2 * w/2
        x = self.head(x)

        ########################################
        x = x +x1 # m* h/2 * w/2
        x11 = self.GST_block(x)

        context3 = interpolation(x11)
        conv2 = interpolation(conv2)
        conv1 = interpolation(conv1)

        x = torch.cat((conv1, conv2, context3), 1)
        x = self.conv_cls(x)

        return x


