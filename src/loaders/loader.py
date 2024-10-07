import numpy as np


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
