import torch
from encoder import TransformerEncoder
from decoder import TransformerDecoder


class TransformerNet(torch.nn.Module):
    """
    Network architecture of the Transformer model.
    """
    def __init__(self, voc_size_enc, voc_size_dec, emb_size, num_heads, hidden_size, dropout, num_layers):
        """
        Constructor for the Transformer model.

        :param voc_size_enc: size of the vocabulary for the encoder
        :param voc_size_dec: size of the vocabulary for the decoder
        :param emb_size: size of the token embeddings
        :param num_heads: number of heads for the multi-head attention
        :param hidden_size: number of neurons in the hidden layer of feed-forward network
        :param dropout: dropout rate
        :param num_layers: number of layers for encoder and decoder
        """
        super(TransformerNet, self).__init__()
        self.encoder = TransformerEncoder(voc_size_enc, emb_size, num_heads, hidden_size, dropout, num_layers)
        self.decoder = TransformerDecoder(voc_size_dec, emb_size, num_heads, hidden_size, dropout, num_layers)
        self.linear_layer = torch.nn.Linear(emb_size, voc_size_dec, bias=False)

    def forward(self, tokens_enc, tokens_dec):
        """
        Forward pass of the Transformer model.

        :param tokens_enc: tokens in input to encoder
        :param tokens_dec: tokens in input to decoder
        :return: the transformer output (logits no probabilities)
        """
        encoder_out = self.encoder(tokens_enc)
        decoder_out = self.decoder(tokens_dec, encoder_out)
        return self.linear_layer(decoder_out)
