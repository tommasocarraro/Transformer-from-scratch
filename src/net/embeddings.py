import torch


class EmbeddingLayer(torch.nn.Module):
    """
    Embedding layer of the Transformer architecture
    """
    def __init__(self, vocab_size, embedding_size):
        """
        Constructor for the Embedding layer.

        :param vocab_size: number of tokens in the vocabulary
        :param embedding_size: number of dimensions in the embedding
        """
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)

    def forward(self, sequence):
        """
        Forward pass of the embedding layer.

        :param sequence: input sequence
        :return: the embeddings of the tokens in the input sequence
        """
        return self.embedding(sequence)


class PositionalEncoding(torch.nn.Module):
    """
    Positional Encoding Module
    """

    def __init__(self, embedding_size, max_len=5000):
        """
        Constructor for the Positional Encoding module.

        :param embedding_size: size of token embeddings
        :param max_len: maximum length of the input sequence
        """
        super(PositionalEncoding, self).__init__()

        # create positional encoding matrix
        pos = torch.arange(max_len).unsqueeze(1).float()
        i = torch.arange(embedding_size).float()
        angle_rates = 1 / torch.pow(10000, i / embedding_size)
        pe = pos * angle_rates
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])

        # add batch dimension for safe summation and ensure pos encoding is not changed during backprop
        self.pos_enc = pe.unsqueeze(0).requires_grad_(False)
        # add the positional encoding to the model and avoid treating it as a parameter
        self.register_buffer('positional_encoding', self.pos_enc)

    def forward(self, sequence_embeddings):
        """
        Forward pass of the positional encoding.

        :param sequence_embeddings: embeddings of the tokens in the input sequence
        :return: embeddings with positional encoding applied
        """
        seq_len = sequence_embeddings.size(1)
        return sequence_embeddings + self.pos_enc[:, :seq_len, :]

