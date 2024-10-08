from src.data.dataset import Dataset
from src.net.transformer import TransformerNet
from src.trainers.trainer import Trainer
from torch.optim import Adam

d = Dataset("./dataset/train.lc.norm.tok.en", "./dataset/train.lc.norm.tok.de", "./dataset/val.lc.norm.tok.en", "./dataset/val.lc.norm.tok.de" )
tr_loader = d.get_loader(128, shuffle=True)
val_loader = d.get_loader(128, shuffle=False, type="val")
transformer = TransformerNet(d.source_vocab_length, d.target_vocab_length, d.padding_token, d.sos_token, d.eos_token, 128, 4, 1028, 0.1, 1)
trainer = Trainer(transformer, Adam(transformer.parameters(), lr=1e-3))
trainer.train(tr_loader, 3, val_loader=val_loader)
print(trainer.infer(val_loader, d.max_seq_length, d.target_vocab, d.padding_token, d.eos_token))
