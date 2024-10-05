import torch
from encoder import TransformerEncoder
from decoder import TransformerDecoder


class TransformerNet(torch.nn.Module):
    """
    Network architecture of the Transformer model.
    """
    def __init__(self, voc_size_enc, voc_size_dec, padding_token_enc, padding_token_dec, emb_size, num_heads,
                 hidden_size, dropout, num_layers):
        """
        Constructor for the Transformer model.

        :param voc_size_enc: size of the vocabulary for the encoder
        :param voc_size_dec: size of the vocabulary for the decoder
        :param padding_token_enc: padding token for the encoder dataset
        :param padding_token_dec: padding token for the decoder dataset
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
        self.padding_token_enc = padding_token_enc
        self.padding_token_dec = padding_token_dec

    def forward(self, tokens_enc, tokens_dec):
        """
        Forward pass of the Transformer model.

        :param tokens_enc: tokens in input to encoder
        :param tokens_dec: tokens in input to decoder
        :return: the transformer output (logits no probabilities)
        """
        # todo different implementation for when inferring, with multiple steps
        # todo manage different sequence length and breack sentences into sub-sentences
        padding_mask_enc = (tokens_enc != self.padding_token_enc).unsqueeze(1).unsqueeze(2)
        padding_mask_dec = (tokens_dec != self.padding_token_dec).unsqueeze(1).unsqueeze(2)
        encoder_out = self.encoder(tokens_enc, padding_mask_enc)
        decoder_out = self.decoder(tokens_dec, encoder_out, padding_mask_enc, padding_mask_dec)
        return self.linear_layer(decoder_out)
