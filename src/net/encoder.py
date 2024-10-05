import torch
from embeddings import EmbeddingLayer, PositionalEncoding
from mh_attention import MultiHeadedAttention
from feed_forward import FeedForward


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
        self.attention = MultiHeadedAttention(num_heads, emb_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(emb_size)

    def forward(self, embeddings):
        """
        Forward pass of the transformer encoder layer.

        :param embeddings: input token embeddings
        :return: transformed token embeddings with attention
        """
        attention_embeddings = self.attention(embeddings)
        attention_embeddings = self.dropout(attention_embeddings)
        intermediate_embeddings = self.layer_norm(embeddings + attention_embeddings)
        feed_forward_embeddings = self.feed_forward(intermediate_embeddings)
        feed_forward_embeddings = self.dropout(feed_forward_embeddings)
        return self.layer_norm(intermediate_embeddings + feed_forward_embeddings)


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

    def forward(self, tokens):
        """
        Forward pass of the transformer encoder.

        :param tokens: input tokens
        :return: encoder output
        """
        embeddings = self.embedding_layer(tokens)
        embeddings = self.positional_encoding(embeddings)
        embeddings = self.dropout(embeddings)
        for layer in self.encoder:
            embeddings = layer(embeddings)
        return embeddings
