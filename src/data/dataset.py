import torch
from torchtext.data import Field
from .utils import tokenize_
from src.loaders.loader import DataLoaderNew, DataLoader
from tokenizers import ByteLevelBPETokenizer


class Dataset:
    """
    Dataset for the translation task.
    """

    def __init__(self, source_lang_file_tr, target_lang_file_tr, source_lang_file_val, target_lang_file_val):
        """
        Constructor for the dataset.

        :param source_lang_file_tr: source language file train
        :param target_lang_file_tr: target language file train
        :param source_lang_file_val: source language file val
        :param target_lang_file_val: target language file val
        """
        # field for tokenization and vocabulary creation
        source_field = Field(tokenize=tokenize_, lower=True, init_token='<SOS>', eos_token='<EOS>')
        target_field = Field(tokenize=tokenize_, lower=True, init_token='<SOS>', eos_token='<EOS>')
        # create vocabularies
        with (open(source_lang_file_tr, 'r', encoding='utf-8') as source_tr,
              open(target_lang_file_tr, 'r', encoding='utf-8') as target_tr,
              open(source_lang_file_val, 'r', encoding='utf-8') as source_val,
              open(target_lang_file_val, 'r', encoding='utf-8') as target_val):
            source_sentences_tr = source_tr.readlines()
            target_sentences_tr = target_tr.readlines()
            source_sentences_val = source_val.readlines()
            target_sentences_val = target_val.readlines()
        # tokenize sentences
        tokenized_source_sentences = [source_field.tokenize(sentence) for sentence in source_sentences_tr]
        tokenized_target_sentences = [target_field.tokenize(sentence) for sentence in target_sentences_tr]
        tokenized_source_sentences_val = [source_field.tokenize(sentence) for sentence in source_sentences_val]
        tokenized_target_sentences_val = [source_field.tokenize(sentence) for sentence in target_sentences_val]
        # build vocabularies on tokenized sentences
        source_field.build_vocab(tokenized_source_sentences + tokenized_source_sentences_val)
        target_field.build_vocab(tokenized_target_sentences + tokenized_target_sentences_val)
        self.source_vocab = source_field.vocab
        self.target_vocab = target_field.vocab
        # get vocab lengths
        self.source_vocab_length = len(source_field.vocab)
        self.target_vocab_length = len(target_field.vocab)
        # get maximum sequence length
        self.max_seq_length = self.get_max_sequence_length(tokenized_source_sentences + tokenized_source_sentences_val,
                                                           tokenized_target_sentences + tokenized_target_sentences_val)
        # preprocess sentences by padding
        self.final_source_sentences_tr = self.process_sentences(source_field, tokenized_source_sentences)
        self.final_target_sentences_tr = self.process_sentences(target_field, tokenized_target_sentences)
        self.final_source_sentences_val = self.process_sentences(source_field, tokenized_source_sentences_val)
        self.final_target_sentences_val = self.process_sentences(target_field, tokenized_target_sentences_val)
        self.padding_token = source_field.vocab.stoi[source_field.pad_token]
        self.eos_token = target_field.vocab.stoi[target_field.eos_token]
        self.sos_token = target_field.vocab.stoi[target_field.init_token]

    @staticmethod
    def get_max_sequence_length(source_sentences, target_sentences):
        """
        It gets the length of the longest sentence across the datasets (source and target).

        :param source_sentences: source sentences
        :param target_sentences: target sentences
        :return: the length of the longest sentence across the datasets (source and target), including special
        SOS and EOS tokens
        """
        source_lengths = [len(sentence) + 2 for sentence in source_sentences]
        target_lengths = [len(sentence) + 2 for sentence in target_sentences]
        return max(max(source_lengths), max(target_lengths))

    def process_sentences(self, field, sentences):
        """
        This function processes the sentences and make them ready for training the model.

        It converts the sentence into its token indexes and add padding tokens to each sentence which is shorter than
        the computed maximum length.

        :param field: field to process the sentences
        :param sentences: sentences to be processed
        :return: processed sentences
        """
        indexed_sentences = [torch.tensor([field.vocab.stoi[field.init_token]] + [
            field.vocab.stoi[token] if token in field.vocab.stoi else field.vocab.stoi[field.unk_token] for token in
            sentence] +
                                          [field.vocab.stoi[field.eos_token]]) for sentence in sentences]
        padded_indexed_sentences = torch.stack([torch.cat([indexed_sentence, torch.full(
            (self.max_seq_length - indexed_sentence.shape[0],), field.vocab.stoi[field.pad_token])], dim=0) if
                                                indexed_sentence.shape[0] < self.max_seq_length else indexed_sentence
                                                for indexed_sentence in indexed_sentences])
        return padded_indexed_sentences

    def get_loader(self, batch_size, shuffle=True, type="train"):
        if type == "train":
            return DataLoader(self.final_source_sentences_tr, self.final_target_sentences_tr, batch_size, shuffle)
        if type == "val":
            return DataLoader(self.final_source_sentences_val, self.final_target_sentences_val, batch_size, shuffle)


class LanguageManager:
    """
    It defines a language manager, with a tokenizer and a vocabulary.
    """

    def __init__(self, language_file, vocab_size=1500, min_freq=2):
        """
        Constructor for the language manager.
        :param language_file: path to file containing sentences.
        :param vocab_size: size of the vocabulary
        :param min_freq: minimum frequency for BPE tokenizer
        """
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train([language_file], vocab_size=vocab_size, min_frequency=min_freq,
                             special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"])
        # field to handle tokenization since tokenizer does not entirely work
        self.field = Field(tokenize=lambda sentence: self.tokenizer.encode(sentence).tokens,
                           lower=False, init_token='<SOS>', eos_token='<EOS>',
                           pad_token='<PAD>', unk_token='<UNK>')
        self.vocab = self.tokenizer.get_vocab()
        self.padding_index = self.vocab[self.field.pad_token]
        self.eos_index = self.vocab[self.field.eos_token]
        self.unk_index = self.vocab[self.field.unk_token]
        self.sos_index = self.vocab[self.field.init_token]

    def process_sentences(self, sentences):
        """
        This function processes the sentences and makes them ready for training the model.

        It converts each sentence into its token indexes and adds padding tokens to fill the max length, if necessary.
        The max length is the length of the longest sentence given to the function.

        :param sentences: sentences to be processed
        :return: processed sentences
        """
        if not isinstance(sentences, list):
            sentences = [sentences]
        tokenized_sentences = [self.field.preprocess(sentence) for sentence in sentences]
        padded_sentences = self.field.pad(tokenized_sentences)
        indexed_sentences = [[self.vocab[token] for token in sentence] for sentence in padded_sentences]
        return indexed_sentences

    def detokenize(self, tokenized_sentences):
        """
        Reconstructs the original sentences (string) from tokenized sentences (list of ids).

        :param tokenized_sentences: A list of tokenized sentences.
        :return: The reconstructed sentences.
        """
        if not isinstance(tokenized_sentences, list):
            tokenized_sentences = [tokenized_sentences]
        return [self.tokenizer.decode(sentence) for sentence in tokenized_sentences]


class DatasetNew:
    """
    Dataset for the translation task.
    """

    def __init__(self, source_lang_file_tr, target_lang_file_tr, source_lang_file_val, target_lang_file_val,
                 source_vocab_size=1500, target_vocab_size=1500, source_min_freq=2, target_min_freq=2):
        """
        Constructor for the dataset.

        :param source_lang_file_tr: source language file train
        :param target_lang_file_tr: target language file train
        :param source_lang_file_val: source language file val
        :param target_lang_file_val: target language file val
        :param source_vocab_size: size of the source vocabulary
        :param target_vocab_size: size of the target vocabulary
        :param source_min_freq: minimum frequency for source BPE tokenizer
        :param target_min_freq: minimum frequency for target BPE tokenizer
        """
        self.source_lang_file_tr = source_lang_file_tr
        self.target_lang_file_tr = target_lang_file_tr
        self.source_lang_file_val = source_lang_file_val
        self.target_lang_file_val = target_lang_file_val
        self.source_lang_manager = LanguageManager(source_lang_file_tr, source_vocab_size, source_min_freq)
        self.target_lang_manager = LanguageManager(target_lang_file_tr, target_vocab_size, target_min_freq)
        self.source_vocab_len = len(self.source_lang_manager.vocab)
        self.target_vocab_len = len(self.target_lang_manager.vocab)
        self.padding_index = self.source_lang_manager.padding_index
        self.eos_index = self.source_lang_manager.eos_index
        self.unk_index = self.source_lang_manager.unk_index
        self.sos_index = self.source_lang_manager.sos_index

    def get_loader(self, batch_size, shuffle=True, type="train"):
        """
        Returns the requested data loader.

        :param batch_size: batch size
        :param shuffle: whether to shuffle the data
        :param type: whether to return train or val loader
        :return: the requested data loader
        """
        if type == "train":
            return DataLoaderNew(self.source_lang_file_tr, self.target_lang_file_tr, self.source_lang_manager,
                                 self.target_lang_manager, batch_size, shuffle)
        if type == "val":
            return DataLoaderNew(self.source_lang_file_val, self.target_lang_file_val, self.source_lang_manager,
                                 self.target_lang_manager, batch_size, shuffle)
