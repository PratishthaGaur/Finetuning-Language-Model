import torch
import torch.nn as nn
from torch.nn import functional as F
import math

##Part - 1
class EncoderHead(nn.Module):

    def __init__(self, head_size,n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
       

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1) * C ** -0.5 
        wei = F.softmax(wei, dim=-1)  
      
        v = self.value(x)
        out = wei @ v 
        return out, wei


class MultiHeadAttentionEncoder(nn.Module):
    def __init__(self, num_heads, head_size,n_embd):
        super().__init__()
        self.heads = nn.ModuleList([EncoderHead(head_size,n_embd) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
    def forward(self, x):
        out_heads = []
        attn_maps = []
        for head in self.heads:
            out, attn_map = head(x)
            out_heads.append(out)
            attn_maps.append(attn_map)
        out = torch.cat(out_heads, dim=-1)
        return out, attn_maps


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)



class BlockEncoder(nn.Module):
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttentionEncoder(n_head, head_size,n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class EncoderBlock(BlockEncoder):
   
    def __init__(self, n_embd, n_head):
        super().__init__(n_embd, n_head)

    def forward(self, x):
        attn_output, attn_maps = self.sa(self.ln1(x))
        x = x + attn_output
        x = x + self.ffwd(self.ln2(x))
        return x, attn_maps

class Encoder(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, n_layer,block_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(block_size, n_embd)
        self.layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, n_embd)
        self.layers = nn.ModuleList([EncoderBlock(n_embd, n_head) for _ in range(n_layer)])
       

    def forward(self, x):
        token_embeddings = self.token_embeddings(x)
        position_embeddings = self.position_embeddings(torch.arange(x.shape[1], device=x.device))
        x = token_embeddings + position_embeddings
        attn_maps = []
        for layer in self.layers:
            x, attn_map = layer(x)
            attn_maps.extend(attn_map)
        x = self.layer_norm(x)
        x = self.lm_head(x) 
       
        return x, attn_maps


##Part -2
class MultiHeadAttentionDecoder(nn.Module):
    def __init__(self, num_heads, head_size,n_embd,block_size):
        super().__init__()
        self.heads = nn.ModuleList([DecoderHead(head_size,n_embd,block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
    def forward(self, x):
        out_heads = []
        attn_maps = []
        for head in self.heads:
            out, attn_map = head(x)
            out_heads.append(out)
            attn_maps.append(attn_map)
        out = torch.cat(out_heads, dim=-1)
        return out, attn_maps


class DecoderHead(nn.Module):

    def __init__(self, head_size,n_embd,block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1) * C ** -0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  
      
        v = self.value(x)
        out = wei @ v 
        return out, wei

class BlockDecoder(nn.Module):

    def __init__(self, n_embd, n_head,block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttentionDecoder(n_head, head_size,n_embd,block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
class DecoderBlock(BlockDecoder):
    def __init__(self, n_embd, n_head,block_size):
        super().__init__(n_embd, n_head,block_size)

    def forward(self, x):
        attn_output, attn_maps = self.sa(self.ln1(x))
        x = x + attn_output
        x = x + self.ffwd(self.ln2(x))
        return x, attn_maps


class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer,block_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([DecoderBlock(n_embd, n_head,block_size) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        token_embeddings = self.token_embeddings(x)
        position_embeddings =  self.position_embeddings(torch.arange(x.shape[1],device=x.device))
        x =  token_embeddings + position_embeddings
        attn_maps = []
        for layer in self.layers:
            x, attn_map = layer(x)
            attn_maps.extend(attn_map)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits, attn_maps
    


##Part-3

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, block_size, n_embd):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.pe = torch.zeros(block_size, n_embd)

        position = torch.arange(0, block_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * -(math.log(10000.0) / n_embd))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.shape[1]]


class RefinedDecoder(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, n_layer,block_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = SinusoidalPositionalEncoding(block_size, n_embd)
        self.layers = nn.ModuleList([DecoderBlock(n_embd, n_head,block_size) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        token_embeddings = self.token_embeddings(x)
        position_embeddings =  self.position_embeddings(x)
        x =  token_embeddings + position_embeddings
        attn_maps = []
        for layer in self.layers:
            x, attn_map = layer(x)
            attn_maps.extend(attn_map)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits, attn_maps