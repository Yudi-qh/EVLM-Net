import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2

import open_clip
from typing import Optional
from transformers import AutoTokenizer,AutoModel,AutoConfig
from open_clip.hf_model import ClsPooler

from urllib.request import urlopen
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms,get_tokenizer
from open_clip.factory import HF_HUB_PREFIX,_MODEL_CONFIGS
import json



def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels*2, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x1, x2):

        x1 = self.up_dwc(x1)
        x1 = channel_shuffle(x1, self.in_channels)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim=1)

        x = self.pwc(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
     
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        # 传入的Hirea Block
        self.block = blk
        dim = blk.attn.qkv.in_features

        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt=self.prompt_learn(x)
        promped=x+prompt
        net=self.block(promped)
        return net


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class BiomedClipTextEncoder(nn.Module):
    def __init__(self):
        super(BiomedClipTextEncoder, self).__init__()
        self.biomedclip=open_clip.create_model('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer=AutoTokenizer.from_pretrained(
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )

    def _forward_bert(self,
                      x,
                      attention_mask:Optional[torch.LongTensor]=None,
                      output_hidden_states: bool=False):
        bert=self.biomedclip.text

        if attention_mask is None:
            attention_mask = (x != bert.config.pad_token_id).long()

        device = next(self.parameters()).device
        x = x.to(device)
        attention_mask = attention_mask.to(device)

        out=bert.transformer(
            input_ids=x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        pooled_out = bert.pooler(out, attention_mask)
        projected = bert.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]

        tokens = (
            out.last_hidden_state[
            :, torch.arange(seq_len) != bert.pooler.cls_token_position, :
            ]
            if type(bert.pooler) == ClsPooler
            else out.last_hidden_state
        )

        if bert.output_tokens:
            return projected, tokens
        if output_hidden_states:
            return projected, out.hidden_states
        else:
            return projected

    def get_conditional_embeddings(
            self,
            batch_size: int,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
    ):
        # compute conditional embeddings from texts
        if len(input_ids) != batch_size:
            raise ValueError(
                "Make sure to pass as many prompt texts as there are query images"
            )
        conditional_embeddings = self._forward_bert(
            input_ids, attention_mask=attention_mask, output_hidden_states=False
        )
        return conditional_embeddings

    def forward(
            self,
            pixel_values: torch.Tensor,
            sentence: list[str],
            # mask_name:list[str],
            # input_ids: torch.Tensor,
            # attention_mask: torch.Tensor,
    ) -> torch.Tensor:


        tokens = self.tokenizer(
            # mask_name,
            sentence,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        conditional_embeddings = self.get_conditional_embeddings(
            batch_size=pixel_values.shape[0],
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return conditional_embeddings


class SAM2UNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2UNet, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        # model=build_sam2(model_cfg)

        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        
        else:
            model = build_sam2(model_cfg)

        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        # print(model.image_encoder.trunk)
        # print(model)
        # print(model.image_encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False

        blocks = []
        for block in self.encoder.blocks:
            # print(block.adapter_attn.parameters())
            if hasattr(block,'adapter_attn'):
                for param in block.adapter_attn.parameters():
                    param.requires_grad=True

            if hasattr(block,'adapter_mlp'):
                for param in block.adapter_mlp.parameters():
                    param.requires_grad=True
            # print(block.adapter_attn.parameters())
            # print(block)
            blocks.append(
                Adapter(block)
            )

        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)

        self.up1 = (EUCB(64, 64))
        self.up2 = (EUCB(64, 64))
        self.up3 = (EUCB(64, 64))
        self.up4 = (EUCB(64, 64))
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

        self.text_encoder=BiomedClipTextEncoder()
        self.gamma_layer1=nn.Linear(512,64)
        self.beta_layer1=nn.Linear(512,64)

        self.gamma_layer2 = nn.Linear(512, 64)
        self.beta_layer2 = nn.Linear(512, 64)

        self.gamma_layer3 = nn.Linear(512, 64)
        self.beta_layer3 = nn.Linear(512, 64)

        self.gamma_layer4 = nn.Linear(512, 64)
        self.beta_layer4 = nn.Linear(512, 64)

    def forward(self, x, sentence=None):

        text_features = self.text_encoder(x, sentence)


        x1, x2, x3, x4 = self.encoder(x)


        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)



        gamma1 = self.gamma_layer1(text_features).unsqueeze(-1).unsqueeze(-1)
        beta1 = self.beta_layer1(text_features).unsqueeze(-1).unsqueeze(-1)
        x1 = gamma1 * x1 + beta1

        gamma2 = self.gamma_layer2(text_features).unsqueeze(-1).unsqueeze(-1)
        beta2 = self.beta_layer2(text_features).unsqueeze(-1).unsqueeze(-1)
        x2 = gamma2 * x2 + beta2


        gamma3 = self.gamma_layer3(text_features).unsqueeze(-1).unsqueeze(-1)
        beta3 = self.beta_layer3(text_features).unsqueeze(-1).unsqueeze(-1)
        x3 = gamma3 * x3 + beta3


        gamma4 = self.gamma_layer4(text_features).unsqueeze(-1).unsqueeze(-1)
        beta4 = self.beta_layer4(text_features).unsqueeze(-1).unsqueeze(-1)
        x4 = gamma4 * x4 + beta4


        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out, out1, out2





