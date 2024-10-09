from src.data.dataset import Dataset
from src.net.transformer import TransformerNet
from src.trainers.trainer import Trainer
from torch.optim import Adam

d = Dataset("./dataset/train.lc.norm.tok.en", "./dataset/train.lc.norm.tok.de", "./dataset/val.lc.norm.tok.en", "./dataset/val.lc.norm.tok.de" )
tr_loader = d.get_loader(128, shuffle=True)
val_loader = d.get_loader(128, shuffle=False, type="val")
transformer = TransformerNet(d.source_vocab_length, d.target_vocab_length, d.padding_token, d.sos_token, d.eos_token, 128, 4, 2048, 0.1, 2, pre_norm=True)
trainer = Trainer(transformer, Adam(transformer.parameters(), lr=1e-3), lr_scheduler=False)
# trainer.train(tr_loader, 100, val_loader=val_loader, early=5, save_path="./transformer.pth")
trainer.load_model("./saved_models/transformer.pth")
print(trainer.infer(["a man sleeping in a green room on a couch ."], d.max_seq_length, d.source_vocab, d.target_vocab,
                    d.padding_token, d.eos_token))
