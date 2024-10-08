import torch
from .embeddings import EmbeddingLayer, PositionalEncoding
from .attention import Attention
from .feed_forward import FeedForward


class EncoderLayer(torch.nn.Module):
    """
    Architecture of the transformer encoder layer.
    """
    def __init__(self, emb_size, num_heads, hidden_size, dropout):
        """
        Constructor for the transformer encoder layer.

        :param emb_size: size of the token embeddings
        :param num_heads: number of heads for the multi-head attention
        :param hidden_size: number of neurons in the hidden layer of the feed-forward network
        :param dropout: dropout rate
        """
        super(EncoderLayer, self).__init__()
        self.feed_forward = FeedForward(emb_size, hidden_size)
        self.attention = Attention(num_heads, emb_size)
        self.dropout = torch.nn.Dropout(dropout)
        # layer normalization aggregates features in different samples (good for NLP)
        # batch normalization aggregates the features of the samples independently (good for CV)
        self.layer_norm_1 = torch.nn.LayerNorm(emb_size)
        self.layer_norm_2 = torch.nn.LayerNorm(emb_size)

    def forward(self, embeddings, padding_mask, pre_norm=False):
        """
        Forward pass of the transformer encoder layer.

        :param embeddings: input token embeddings
        :param padding_mask: mask to avoid the padding tokens to be included in the attention computation
        :param pre_norm: whether to apply layer normalization before sublayer
        :return: transformed token embeddings with attention
        """
        if pre_norm:
            norm_embeddings = self.layer_norm_1(embeddings)
            attention_embeddings = self.attention(norm_embeddings, norm_embeddings, norm_embeddings,
                                                  padding_mask=padding_mask)
            attention_embeddings = self.dropout(attention_embeddings)
            # here I sum normalized embeddings
            intermediate_embeddings = norm_embeddings + attention_embeddings
            norm_intermediate_embeddings = self.layer_norm_2(intermediate_embeddings)
            feed_forward_embeddings = self.feed_forward(norm_intermediate_embeddings)
            feed_forward_embeddings = self.dropout(feed_forward_embeddings)
            # here, I sum normalized embeddings
            return norm_intermediate_embeddings + feed_forward_embeddings
        else:
            attention_embeddings = self.attention(embeddings, embeddings, embeddings, padding_mask=padding_mask)
            attention_embeddings = self.dropout(attention_embeddings)
            intermediate_embeddings = self.layer_norm_1(embeddings + attention_embeddings)
            feed_forward_embeddings = self.feed_forward(intermediate_embeddings)
            feed_forward_embeddings = self.dropout(feed_forward_embeddings)
            return self.layer_norm_2(intermediate_embeddings + feed_forward_embeddings)


class TransformerEncoder(torch.nn.Module):
    """
    Transformer encoder architecture.
    """
    def __init__(self, voc_size, emb_size, num_heads, hidden_size, dropout, num_layers):
        """
        Constructor for the transformer encoder architecture.

        :param voc_size: size of the vocabulary
        :param emb_size: size of the token embeddings
        :param num_heads: number of heads for the multi-head attention
        :param hidden_size: number of neurons in the hidden layer of the feed-forward network
        :param dropout: dropout rate
        :param num_layers: number of layers in the encoder
        """
        super(TransformerEncoder, self).__init__()
        self.embedding_layer = EmbeddingLayer(voc_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size)
        self.encoder = torch.nn.ModuleList([EncoderLayer(emb_size, num_heads, hidden_size, dropout)
                                            for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, tokens, padding_mask):
        """
        Forward pass of the transformer encoder.

        :param tokens: input tokens
        :param padding_mask: mask to avoid the padding tokens to be included in the attention computation
        :return: encoder output
        """
        embeddings = self.embedding_layer(tokens)
        embeddings = self.positional_encoding(embeddings)
        embeddings = self.dropout(embeddings)
        for layer in self.encoder:
            embeddings = layer(embeddings, padding_mask)
        return embeddings
