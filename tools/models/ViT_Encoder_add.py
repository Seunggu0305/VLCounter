import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch import nn

from functools import reduce
from operator import mul

import math
from .Encoder_utils import LayerNorm, Transformer, Attention


class SPTCLIPVisionTransformer(nn.Module):
    def __init__(self, input_resolution=384, patch_size=16, width=768, layers=12, heads=12, output_dim=512, drop_path_rate=0.1, out_indices=[5,6,7,8,11], pretrained=None, get_embeddings=True, 
                 num_tokens=10, prompt_dim=768, total_d_layer=11, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings
        self.num_layers = layers

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.out_indices = out_indices

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.text_proj = nn.Linear(512, width)
        nn.init.kaiming_normal_(self.text_proj.weight, a=0, mode='fan_out') 
        self.text_dropout = nn.Dropout(0.1)
        ## Setting of visual prompt tuning
        self.num_tokens = num_tokens 
        self.prompt_dim = prompt_dim
        self.total_d_layer = total_d_layer


        ## Add the prompt parameters # exclude_key=prompt:
        self._init_prompt(patch_size, self.num_tokens, self.prompt_dim, self.total_d_layer)

        self.embed_dim = width
        self.num_heads = heads
        self.patch_size = patch_size
        
    
    def _init_prompt(self, patch, num_tokens, prompt_dim, total_d_layer):
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        if total_d_layer >= 0:
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if total_d_layer > 0:  # noqa
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = Dropout(0.1)

        else: # total_d_layer < 0
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(abs(total_d_layer), num_tokens, prompt_dim))
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = Dropout(0.1)
            

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
            # checkpoint = torch.load(pretrained)['model']

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                # if k.startswith('module.visual.'):
                #     new_k = k.replace('module.visual.', '')
                    state_dict[new_k] = checkpoint[k].float()

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    # (1025, 768)                      (197, 768)  
                    print(f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]
                    if self.patch_size == 16:
                        spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    elif self.patch_size == 32:
                        spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 7, 7, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    else:
                        assert AttributeError('Patch Size should be 16 or 32')
                    spatial_pos = spatial_pos.reshape(768, self.spatial_size*self.spatial_size).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape
            
            # del state_dict['conv1.weight']
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')

            self.attn = None
            if self.attn == None:
                for i in range(1,2): # surgery 7, maskclip 2
                    self.attn = Attention(self.embed_dim, self.embed_dim, self.num_heads, True)
                    self.attn.qkv.weight.data = self.transformer.resblocks[-i].attn.in_proj_weight.clone()
                    self.attn.qkv.bias.data = self.transformer.resblocks[-i].attn.in_proj_bias.clone()
                    self.attn.proj.weight.data = self.transformer.resblocks[-i].attn.out_proj.weight.clone()
                    self.attn.proj.bias.data = self.transformer.resblocks[-i].attn.out_proj.bias.clone()
                    self.transformer.resblocks[-i].attn = self.attn

    def forward(self, x: torch.Tensor, _t: torch.Tensor):
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        
        if self.total_d_layer >=0:
            # concat prompt
            x = torch.cat((
                x[:, :1, :],
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)) + self.text_dropout(self.text_proj(_t).expand(-1, self.num_tokens, -1)),
                    x[:, 1:, :]
                ), dim=1)

        x = x.permute(1, 0, 2)

        features = []
        outs = []
        x, features = self.forward_deep_prompt(x, features, H, W, _t)

        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, 0]
            visual_embedding = x[:, -(H*W):].reshape(B, H, W, -1).permute(0, 3, 1, 2)

            visual_embedding = visual_embedding / visual_embedding.norm(dim=1, keepdim=True)
            features.append(visual_embedding)

            outs.append(tuple(features))
            global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True)
            outs.append(global_embedding)

        return outs[0]


    def forward_deep_prompt(self, embedding_output, features, H, W, _t, out_last=False):
        B = embedding_output.shape[1]

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
            elif i <= self.deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2)
                deep_text = self.text_dropout(self.text_proj(_t).expand(-1,self.num_tokens,-1)).permute(1,0,2)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb + deep_text,
                    hidden_states[(1+self.num_tokens):, :, :]
                ), dim=0)

                hidden_states = self.transformer.resblocks[i](hidden_states)
            else:
                hidden_states = self.transformer.resblocks[i](hidden_states)
            
            if len(self.out_indices) > 1:
                if i in self.out_indices[:-1]:
                    xp = hidden_states.permute(1, 0, 2)[:, -(H*W):, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous() / xp.norm(dim=1,keepdim=True))

        encoded = self.prompt_norm(hidden_states)
        return encoded, features