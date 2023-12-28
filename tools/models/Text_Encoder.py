import torch
import torch.nn as nn

from .Encoder_utils import LayerNorm, Transformer


"""Text Encoder"""

class CLIPTextEncoder(nn.Module):
    def __init__(self, context_length=77,
                 vocab_size=49408,
                #  vocab_size=49408+1,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=512,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.text_projection = nn.Linear(transformer_width, embed_dim)
    
    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
            # checkpoint = torch.load(pretrained)['model']

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                # if k.startswith('module.encode_text.transformer.'):
                #     new_k = k.replace('module.encode_text.', '')
                    # state_dict[new_k] = checkpoint[k].float()
                    state_dict[k] = checkpoint[k].float()
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                # if k == 'module.encode_text.positional_embedding' or k.startswith('module.encode_text.text_projection') or k.startswith('module.encode_text.token_embedding') or k.startswith('module.encode_text.ln_final'):
                #     new_k = k.replace('module.encode_text.', '')
                    # if new_k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    # state_dict[new_k] = checkpoint[k]
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            if u != [] or w != [] :
                print(u, w, 'are misaligned params in text encoder')


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding 
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # x = self.text_projection(x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        return x