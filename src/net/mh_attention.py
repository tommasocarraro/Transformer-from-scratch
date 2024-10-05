import math

import torch


class MultiHeadedAttention(torch.nn.Module):
    """
    Multi-headed attention module
    """
    def __init__(self, num_heads, embedding_size):
        """
        Constructor of the MultiHeadedAttention module

        :param num_heads: number of heads of the multi-headed attention
        :param embedding_size: size of the token embeddings
        """
        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        assert embedding_size % num_heads == 0, "embedding_size must be divisible by num_heads"
        # linear projections for Q, K, and V
        self.wq = torch.nn.Linear(embedding_size, embedding_size, bias=False)
        self.wk = torch.nn.Linear(embedding_size, embedding_size, bias=False)
        self.wv = torch.nn.Linear(embedding_size, embedding_size, bias=False)
        # linear projection for the output
        self.wo = torch.nn.Linear(embedding_size, embedding_size, bias=False)

    def forward(self, input_embeddings, mask=False, input_embeddings_cross_attention=None):
        """
        Forward of the Multi-headed attention module

        :param input_embeddings: input embedding tensor
        :param mask: whether to mask the attention weights (used for the decoder)
        :param input_embeddings_cross_attention: input embedding tensor for cross attention
        :return: output of the Multi-headed attention module (special embeddings)
        """
        # compute linear projections
        if input_embeddings_cross_attention is None:
            q_prime = self.wq(input_embeddings)
            k_prime = self.wk(input_embeddings)
            v_prime = self.wv(input_embeddings)
        else:
            q_prime = self.wq(input_embeddings_cross_attention)
            k_prime = self.wk(input_embeddings)
            v_prime = self.wv(input_embeddings)

        # split matrices into different heads before applying attention
        q_prime = q_prime.view(-1, self.num_heads, q_prime.shape[1], self.embedding_size // self.num_heads)
        k_prime = k_prime.view(-1, self.num_heads, k_prime.shape[1], self.embedding_size // self.num_heads)
        v_prime = v_prime.view(-1, self.num_heads, v_prime.shape[1], self.embedding_size // self.num_heads)

        # compute the attentions of the different heads in a single step
        qk = torch.matmul(q_prime, k_prime.permute(0, 1, 3, 2)) / math.sqrt(self.embedding_size // self.num_heads)
        # auto-regressive mask
        if mask:
            attention_mask = torch.tril(torch.ones(q_prime.shape[2], q_prime.shape[2]))
            qk = qk.masked_fill(attention_mask == 0, float('-inf'))
        # compute the softmax
        qk_softmax = torch.nn.functional.softmax(qk, dim=-1)
        # compute the result of attention
        qkv = torch.matmul(qk_softmax, v_prime)

        # compute final linear projection
        return self.wo(qkv.view(-1, q_prime.shape[2], self.embedding_size))
