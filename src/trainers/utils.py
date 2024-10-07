import torch


def tokens_to_sentences(token_sequences, target_vocab, padding_token, eos_token):
    """
    Converts token indices to readable sentences using the decoder vocabulary.

    :param token_sequences: Tensor containing token indices
    :param target_vocab: Target vocabulary
    :param padding_token: Padding token
    :param eos_token: EOS token
    :return: List of sentences in the target language
    """
    sentences = []
    for seq in token_sequences:
        # Convert the sequence of token indices to words
        words = [target_vocab.itos[token.item()] for token in seq if
                 token.item() != padding_token and token.item() != eos_token]
        sentences.append(' '.join(words))  # Join words to form a sentence
    return sentences


def get_device():
    """
    Get the device on which the experiments have to be run.
    :return: the device on which the experiments have to be run.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")  # Use the first GPU
    else:
        return torch.device("cpu")
