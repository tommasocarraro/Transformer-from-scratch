import torch
from tqdm import tqdm
from .utils import tokens_to_sentences
from src import get_device


class Trainer:
    """
    Generic trainer for the translation task.
    """
    def __init__(self, transformer_model, optimizer):
        """
        Initialize the trainer with the given transformer model.

        :param transformer_model: transformer model
        """
        self.transformer_model = transformer_model.to(get_device())
        self.optimizer = optimizer
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=self.transformer_model.padding_token)

    def train(self, train_loader, n_epochs, val_loader=None, verbose=1):
        """
        Train the transformer model.

        :param train_loader: loader for the training data
        :param n_epochs: number of epochs for training
        :param val_loader: loader for the validation data
        :param verbose: logging verbosity
        """
        self.transformer_model.train()
        for epoch in range(n_epochs):
            # training step
            train_loss = self.train_epoch(train_loader, epoch + 1)
            # validation step
            val_score = self.validate(val_loader)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                print("Epoch %d - Train loss %.3f - Validation CE_Loss %.3f"
                      % (epoch + 1, train_loss, val_score))
            # save best model and update early stop counter, if necessary
            # if val_score > best_val_score:
            #     best_val_score = val_score
            #     early_counter = 0
            #     if save_path:
            #         self.save_model(save_path)
            # else:
            #     early_counter += 1
            #     if early is not None and early_counter > early:
            #         print("Training interrupted due to early stopping")
            #         break

    def validate(self, val_loader):
        self.transformer_model.eval()
        val_loss = 0.0
        for batch_idx, (source_sentences, target_sentences) in enumerate(val_loader):
            with torch.no_grad():
                preds = self.transformer_model(source_sentences.to(get_device()), target_sentences.to(get_device()))
                loss = self.cross_entropy_loss(preds.view(-1, preds.shape[-1]),
                                               target_sentences.view(-1).to(get_device()))
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def train_epoch(self, train_loader, epoch_idx):
        """
        Method for the training of one single epoch.

        :param train_loader: loader for the training data
        :param epoch_idx: index of the current epoch
        :return: loss function averaged over all batches
        """
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Batches of epoch %d" % (epoch_idx, ), unit="batch")
        for batch_idx, (source_sentences, target_sentences) in enumerate(progress_bar):
            self.optimizer.zero_grad()
            preds = self.transformer_model(source_sentences.to(get_device()), target_sentences.to(get_device()))
            loss = self.cross_entropy_loss(preds.view(-1, preds.shape[-1]), target_sentences.view(-1).to(get_device()))
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({"CE_Loss": loss.item()})
        return train_loss / len(train_loader)

    def infer(self, inference_loader, max_seq_len, target_vocab, padding_token, eos_token):
        """
        Method for inference.

        :param inference_loader: loader containing sentences that have to be translated
        :param max_seq_len: maximum length of the sentence
        :param target_vocab: target vocabulary
        :param padding_token: padding token
        :param eos_token: eos token
        :return: translated sentences
        """
        self.transformer_model.eval()
        predictions = []
        for batch_idx, (source_sentences, _) in enumerate(inference_loader):
            preds = self.transformer_model.infer(source_sentences.to(get_device()), max_seq_len)
            predictions.append(preds)
        return tokens_to_sentences(torch.cat(predictions), target_vocab, padding_token, eos_token)
