import math
from src import get_device
import torch


class Attention(torch.nn.Module):
    """
    Attention module
    """
    def __init__(self, num_heads, embedding_size):
        """
        Constructor of the Attention module

        :param num_heads: number of heads of the multi-headed attention
        :param embedding_size: size of the token embeddings
        """
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        assert embedding_size % num_heads == 0, "embedding_size must be divisible by num_heads"
        # linear projections for Q, K, and V
        self.wq = torch.nn.Linear(embedding_size, embedding_size, bias=False)
        self.wk = torch.nn.Linear(embedding_size, embedding_size, bias=False)
        self.wv = torch.nn.Linear(embedding_size, embedding_size, bias=False)
        # linear projection for the output
        self.wo = torch.nn.Linear(embedding_size, embedding_size, bias=False)

    def forward(self, q, k, v, autoregressive_mask=False, padding_mask=None):
        """
        Forward of the Multi-headed attention module

        :param q: query matrix
        :param k: key matrix
        :param v: value matrix
        :param autoregressive_mask: whether to mask the attention weights (used for the autoregressive decoder)
        :param padding_mask: padding mask to avoid computing attention for padding tokens
        :return: output of the Multi-headed attention module (special embeddings)
        """
        # compute linear projections
        q_prime = self.wq(q)
        k_prime = self.wk(k)
        v_prime = self.wv(v)

        # split matrices into different heads before applying attention
        q_prime = q_prime.view(-1, self.num_heads, q_prime.shape[1], self.embedding_size // self.num_heads)
        k_prime = k_prime.view(-1, self.num_heads, k_prime.shape[1], self.embedding_size // self.num_heads)
        v_prime = v_prime.view(-1, self.num_heads, v_prime.shape[1], self.embedding_size // self.num_heads)

        # compute the attentions of the different heads in a single step
        qk = torch.matmul(q_prime, k_prime.permute(0, 1, 3, 2)) / math.sqrt(self.embedding_size // self.num_heads)

        # auto-regressive mask
        if autoregressive_mask:
            attention_mask = torch.tril(torch.ones(q_prime.shape[2], q_prime.shape[2])).to(get_device())
            qk = qk.masked_fill(attention_mask == 0, -1e9)

        # padding mask
        if padding_mask is not None:
            qk = qk.masked_fill(padding_mask == 1, -1e9)

        # compute the softmax
        qk_softmax = torch.nn.functional.softmax(qk, dim=-1)

        # compute the result of attention
        qkv = torch.matmul(qk_softmax, v_prime)

        # compute final linear projection
        return self.wo(qkv.view(-1, q_prime.shape[2], self.embedding_size))
