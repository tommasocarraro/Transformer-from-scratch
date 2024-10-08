import torch
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from src import get_device


class TransformerNet(torch.nn.Module):
    """
    Network architecture of the Transformer model.
    """
    def __init__(self, voc_size_enc, voc_size_dec, padding_token, sos_token, eos_token, emb_size, num_heads,
                 hidden_size, dropout, num_layers, pre_norm=False):
        """
        Constructor for the Transformer model.

        :param voc_size_enc: size of the vocabulary for the encoder
        :param voc_size_dec: size of the vocabulary for the decoder
        :param padding_token: padding token index (shared across encoder and decoder datasets)
        :param sos_token: sos token index
        :param eos_token: eos token index
        :param emb_size: size of the token embeddings
        :param num_heads: number of heads for the multi-head attention
        :param hidden_size: number of neurons in the hidden layer of feed-forward network
        :param dropout: dropout rate
        :param num_layers: number of layers for encoder and decoder
        :param pre_norm: whether to use pre-normalization instead of post-normalization for layer normalization
        """
        super(TransformerNet, self).__init__()
        self.encoder = TransformerEncoder(voc_size_enc, emb_size, num_heads, hidden_size, dropout, num_layers)
        self.decoder = TransformerDecoder(voc_size_dec, emb_size, num_heads, hidden_size, dropout, num_layers)
        self.linear_layer = torch.nn.Linear(emb_size, voc_size_dec, bias=False)
        self.padding_token = padding_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pre_norm = pre_norm

    def forward(self, tokens_enc, tokens_dec):
        """
        Forward pass of the Transformer model.

        :param tokens_enc: tokens in input to encoder
        :param tokens_dec: tokens in input to decoder
        :return: the transformer output (logits no probabilities)
        """
        padding_mask_enc = (tokens_enc == self.padding_token).unsqueeze(1).unsqueeze(2).to(get_device())
        padding_mask_dec = (tokens_dec == self.padding_token).unsqueeze(1).unsqueeze(2).to(get_device())
        encoder_out = self.encoder(tokens_enc, padding_mask_enc, pre_norm=self.pre_norm)
        decoder_out = self.decoder(tokens_dec, encoder_out, padding_mask_enc, padding_mask_dec, pre_norm=self.pre_norm)
        return self.linear_layer(decoder_out)

    def infer(self, tokens_enc):
        """
        It performs the prediction of the Transformer model given a source language sentence.

        :param tokens_enc: tokens in input to encoder
        :return: the predicted tokens in the target language
        """
        # get encoder output for the given source sentences
        padding_mask_enc = (tokens_enc == self.padding_token).unsqueeze(1).unsqueeze(2).to(get_device())
        encoder_output = self.encoder(tokens_enc, padding_mask_enc, pre_norm=self.pre_norm)

        # initialize the input for the decoder with SOS tokens
        batch_size = tokens_enc.shape[0]
        decoder_input = torch.full((batch_size, 1), self.sos_token).to(get_device())

        # Store parallel generated sequences
        generated_sequences = []
        eos_check = torch.zeros(batch_size, 1, dtype=torch.bool).to(get_device())

        for _ in range(tokens_enc.shape[1]):
            decoder_output = self.decoder(decoder_input, encoder_output, padding_mask_enc=padding_mask_enc,
                                          pre_norm=self.pre_norm)
            decoder_output = self.linear_layer(decoder_output)
            next_token = decoder_output[:, -1, :].argmax(dim=-1, keepdim=True)
            decoder_input = torch.cat((decoder_input, next_token), dim=1)
            generated_sequences.append(next_token)

            eos_check = eos_check | (next_token.squeeze() == self.eos_token)

            if eos_check.all():
                break  # stop if all sequences generated EOS token

        generated_sequences = torch.cat(generated_sequences, dim=1)
        return generated_sequences
