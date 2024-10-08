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
    for batch in token_sequences:
        for seq in batch:
            # Convert the sequence of token indices to words
            words = [target_vocab.itos[token.item()] for token in seq if token.item() != padding_token]
            words = words[:words.index(eos_token)] if eos_token in words else words
            sentences.append(' '.join(words))  # Join words to form a sentence
    return sentences
