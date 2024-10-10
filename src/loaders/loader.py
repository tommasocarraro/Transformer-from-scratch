import numpy as np
import torch
from src import get_device


class DataLoader:
    """
    Generic data loader for the translation task.
    """
    def __init__(self, source_sentences, target_sentences=None, batch_size=128, shuffle=True):
        """
        Constructor for the data loader.

        :param source_sentences: sentences in the source language
        :param batch_size: corresponding sentences in the target language
        :param shuffle: whether to shuffle the data
        """
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.source_sentences) / self.batch_size))

    def __iter__(self):
        n = self.source_sentences.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            batch_source_sentences = self.source_sentences[idxlist[start_idx:end_idx]]
            if self.target_sentences is not None:
                batch_target_sentences = self.target_sentences[idxlist[start_idx:end_idx]]
                yield batch_source_sentences, batch_target_sentences
            else:
                yield batch_source_sentences


class DataLoaderNew:
    """
    Generic data loader for the translation task.
    """
    def __init__(self, source_lang_file, target_lang_file, source_lang_manager, target_lang_manager, batch_size=128,
                 shuffle=True):
        """
        Constructor for the data loader.

        :param source_lang_file: path to the source language file
        :param target_lang_file: path to the target language file
        :param source_lang_manager: source language manager to tokenize source language sentences
        :param target_lang_manager: target language manager to tokenize target language sentences
        :param batch_size: batch size of the data loader
        :param shuffle: whether to shuffle the data
        """
        with open(source_lang_file, 'r') as s, open(target_lang_file, 'r') as t:
            self.source_sentences = s.readlines()
            self.target_sentences = t.readlines()
        self.source_lang_manager = source_lang_manager
        self.target_lang_manager = target_lang_manager
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.source_sentences) / self.batch_size))

    def __iter__(self):
        n = len(self.source_sentences)
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            batch_source_sentences = [self.source_sentences[i] for i in idxlist[start_idx:end_idx]]
            batch_target_sentences = [self.target_sentences[i] for i in idxlist[start_idx:end_idx]]
            tokenized_source_sentences = self.source_lang_manager.process_sentences(batch_source_sentences)
            tokenized_target_sentences = self.target_lang_manager.process_sentences(batch_target_sentences)
            yield (torch.tensor(tokenized_source_sentences).to(get_device()),
                   torch.tensor(tokenized_target_sentences).to(get_device()))
