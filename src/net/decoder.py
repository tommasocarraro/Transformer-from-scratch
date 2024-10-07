import torch
from .embeddings import EmbeddingLayer, PositionalEncoding
from .attention import Attention
from .feed_forward import FeedForward


class DecoderLayer(torch.nn.Module):
    """
    Architecture of the transformer decoder layer.
    """

    def __init__(self, emb_size, num_heads, hidden_size, dropout):
        """
        Constructor for the transformer decoder layer.

        :param emb_size: size of the token embeddings
        :param num_heads: number of heads for the multi-head attention
        :param hidden_size: number of neurons in the hidden layer of the feed-forward network
        :param dropout: dropout rate
        """
        super(DecoderLayer, self).__init__()
        self.feed_forward = FeedForward(emb_size, hidden_size)
        self.self_attention = Attention(num_heads, emb_size)
        self.cross_attention = Attention(num_heads, emb_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm_1 = torch.nn.LayerNorm(emb_size)
        self.layer_norm_2 = torch.nn.LayerNorm(emb_size)
        self.layer_norm_3 = torch.nn.LayerNorm(emb_size)

    def forward(self, embeddings, encoder_outputs, padding_mask_enc, padding_mask_dec=None):
        """
        Forward pass of the transformer decoder layer.

        :param embeddings: input token embeddings
        :param encoder_outputs: encoder outputs
        :param padding_mask_enc: padding mask to avoid the padding token to be included in the attention computation
        of the encoder
        :param padding_mask_dec: padding mask to avoid the padding token to be included in the attention computation of
        the decoder
        :return: transformed token embeddings with attention
        """
        # masked attention
        attention_embeddings = self.self_attention(embeddings, embeddings, embeddings,
                                                   autoregressive_mask=True, padding_mask=padding_mask_dec)
        attention_embeddings = self.dropout(attention_embeddings)
        intermediate_embeddings = self.layer_norm_1(embeddings + attention_embeddings)
        # attention with key and values coming from encoder and queries from decoder
        intermediate_attention_embeddings = self.cross_attention(q=intermediate_embeddings, k=encoder_outputs,
                                                                 v=encoder_outputs, padding_mask=padding_mask_enc)
        intermediate_attention_embeddings = self.dropout(intermediate_attention_embeddings)
        pre_feed_forward = self.layer_norm_2(intermediate_embeddings + intermediate_attention_embeddings)
        # feed-forward network
        feed_forward_embeddings = self.feed_forward(pre_feed_forward)
        feed_forward_embeddings = self.dropout(feed_forward_embeddings)
        return self.layer_norm_3(pre_feed_forward + feed_forward_embeddings)


class TransformerDecoder(torch.nn.Module):
    """
    Transformer decoder architecture.
    """

    def __init__(self, voc_size, emb_size, num_heads, hidden_size, dropout, num_layers):
        """
        Constructor for the transformer decoder architecture.

        :param voc_size: size of the vocabulary
        :param emb_size: size of the token embeddings
        :param num_heads: number of heads for the multi-head attention
        :param hidden_size: number of neurons in the hidden layer of the feed-forward network
        :param dropout: dropout rate
        :param num_layers: number of layers in the encoder
        """
        super(TransformerDecoder, self).__init__()
        self.embedding_layer = EmbeddingLayer(voc_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size)
        self.decoder = torch.nn.ModuleList([DecoderLayer(emb_size, num_heads, hidden_size, dropout)
                                            for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, tokens, encoder_outputs, padding_mask_enc, padding_mask_dec=None):
        """
        Forward pass of the transformer decoder.

        :param tokens: input tokens
        :param encoder_outputs: encoder outputs
        :param padding_mask_enc: padding mask to avoid the padding token to be included in the attention computation
        of the encoder
        :param padding_mask_dec: padding mask to avoid the padding token to be included in the attention computation of
        the decoder
        :return: encoder output
        """
        embeddings = self.embedding_layer(tokens)
        embeddings = self.positional_encoding(embeddings)
        embeddings = self.dropout(embeddings)
        for layer in self.decoder:
            embeddings = layer(embeddings, encoder_outputs, padding_mask_enc, padding_mask_dec)
        return embeddings
