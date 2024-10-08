import torch
from tqdm import tqdm
from .utils import tokens_to_sentences
from src import get_device
from torch.optim.lr_scheduler import LambdaLR


class Trainer:
    """
    Generic trainer for the translation task.
    """

    def __init__(self, transformer_model, optimizer, lr_scheduler=False):
        """
        Initialize the trainer with the given transformer model.

        :param transformer_model: transformer model
        :param optimizer: optimizer
        :param scheduler: lr scheduler
        """
        self.transformer_model = transformer_model.to(get_device())
        self.optimizer = optimizer
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=self.transformer_model.padding_token)
        self.lr_scheduler = lr_scheduler
        if lr_scheduler:
            def lr_lambda(step):
                """
                Learning rate scheduler as proposed in the original paper.

                :param step: step number
                :return: new learning rate value
                """
                warmup_steps = 4000
                d_model = 512  # or any other model size
                step = max(1, step)
                return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)

            self.scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    def train(self, train_loader, n_epochs, val_loader=None, verbose=1, early=None, save_path=None):
        """
        Train the transformer model.

        :param train_loader: loader for the training data
        :param n_epochs: number of epochs for training
        :param val_loader: loader for the validation data
        :param verbose: logging verbosity
        :param early: number of epochs for early stopping. If it is not None, early stopping is performed and at the
        end of the training the best model weights are loaded
        :param save_path: path to save the model
        """
        early_counter, best_val_score = 0, 0
        for epoch in range(n_epochs):
            # training step
            train_loss, train_acc = self.train_epoch(train_loader, epoch + 1)
            # validation step
            val_score = self.validate(val_loader)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                print("Epoch %d - Train loss %.3f - Train acc %.3f - Validation CE_Loss %.3f - Validation acc %.3f"
                      % (epoch + 1, train_loss, train_acc, val_score[0], val_score[1]))
            # save best model and update early stop counter, if necessary
            if val_score > best_val_score:
                best_val_score = val_score
                early_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    print("Training interrupted due to early stopping")
                    print("Loading best weights in the model")
                    self.load_model(save_path)
                    break

    def save_model(self, save_path):
        """
        Save the model to the given path.

        :param save_path: path to save the model
        """
        torch.save({
            'model_state_dict': self.transformer_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)

    def load_model(self, path):
        """
        Method for loading the model.

        :param path: path from which the model has to be loaded.
        """
        checkpoint = torch.load(path)
        self.transformer_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def validate(self, val_loader):
        self.transformer_model.eval()
        val_loss, val_acc = 0.0, 0.0
        for batch_idx, (source_sentences, target_sentences) in enumerate(val_loader):
            with torch.no_grad():
                preds = self.transformer_model(source_sentences.to(get_device()),
                                               target_sentences[:, :-1].to(get_device()))
                loss = self.cross_entropy_loss(preds.view(-1, preds.shape[-1]),
                                               target_sentences[:, 1:].reshape(-1).to(get_device()))
                val_acc += self.calculate_accuracy(preds.view(-1, preds.shape[-1]),
                                                   target_sentences[:, 1:].reshape(-1).to(get_device()))
                val_loss += loss.item()
        return val_loss / len(val_loader), val_acc / len(val_loader)

    def train_epoch(self, train_loader, epoch_idx):
        """
        Method for the training of one single epoch.

        :param train_loader: loader for the training data
        :param epoch_idx: index of the current epoch
        :return: loss function averaged over all batches
        """
        self.transformer_model.train()
        train_loss, train_acc = 0.0, 0.0
        progress_bar = tqdm(train_loader, desc="Batches of epoch %d" % (epoch_idx,), unit="batch")
        for batch_idx, (source_sentences, target_sentences) in enumerate(progress_bar):
            self.optimizer.zero_grad()
            preds = self.transformer_model(source_sentences.to(get_device()), target_sentences[:, :-1].to(get_device()))
            loss = self.cross_entropy_loss(preds.view(-1, preds.shape[-1]),
                                           target_sentences[:, 1:].reshape(-1).to(get_device()))
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            acc = self.calculate_accuracy(preds.view(-1, preds.shape[-1]),
                                                 target_sentences[:, 1:].reshape(-1).to(get_device()))
            train_acc += acc
            progress_bar.set_postfix({"CE_Loss": loss.item()})
            progress_bar.set_postfix({"Train_Acc": acc})
        if self.lr_scheduler:
            self.scheduler.step()
        return train_loss / len(train_loader), train_acc / len(train_loader)

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

    def calculate_accuracy(self, preds, target_sentences):
        """
        Calculate multiclass accuracy by comparing predicted tokens to target tokens.

        :param preds: model predictions (batch_size, seq_len, vocab_size)
        :param target_sentences: ground truth target sentences (batch_size, seq_len)
        :return: accuracy score (percentage of correctly predicted tokens)
        """
        preds = preds.argmax(dim=-1)
        # create mask to mask padding
        mask = target_sentences != self.transformer_model.padding_token
        # compute masked accuracy (only for non-padded tokens)
        correct_predictions = (preds == target_sentences) & mask
        correct_tokens = correct_predictions.sum().item()
        # the sum only includes non-padded tokens
        total_tokens = mask.sum().item()
        return correct_tokens / total_tokens if total_tokens > 0 else 0.0
