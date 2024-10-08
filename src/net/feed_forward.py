import torch


class FeedForward(torch.nn.Module):
    """
    Feed-forward network applied after attention in the Transformer architecture.
    """
    def __init__(self, embedding_size=512, hidden_size=2048):
        """
        Constructor for the feed-forward network.

        :param embedding_size: size of the token embeddings.
        :param hidden_size: size of the hidden layer.
        """
        super(FeedForward, self).__init__()
        self.layer_1 = torch.nn.Linear(embedding_size, hidden_size)
        self.layer_2 = torch.nn.Linear(hidden_size, embedding_size)
        self.relu = torch.nn.ReLU()

    def forward(self, attention_embeddings):
        """
        Forward pass of the feed-forward network.

        :param attention_embeddings: embeddings of the tokens after attention.
        :return: embeddings after applying non-linear transformation.
        """
        out = self.relu(self.layer_1(attention_embeddings))
        return self.layer_2(out)
