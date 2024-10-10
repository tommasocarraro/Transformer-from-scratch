from src.data.dataset import DatasetNew, LanguageManager
from src.net.transformer import TransformerNet
from src.trainers.trainer import Trainer
from torch.optim import Adam

d = DatasetNew("./dataset/train.lc.norm.tok.en", "./dataset/train.lc.norm.tok.de", "./dataset/val.lc.norm.tok.en", "./dataset/val.lc.norm.tok.de" )
tr_loader = d.get_loader(128, shuffle=True)
val_loader = d.get_loader(128, shuffle=False, type="val")
transformer = TransformerNet(d.source_vocab_len, d.target_vocab_len, d.padding_index, d.sos_index, d.eos_index,
                             128, 4, 2048, 0.1, 2, pre_norm=True)
trainer = Trainer(transformer, Adam(transformer.parameters(), lr=1e-3), lr_scheduler=False)
# trainer.train(tr_loader, 100, val_loader=val_loader, early=5, save_path="./transformer.pth")
trainer.load_model("./transformer.pth")
print(trainer.infer(["a man sleeping in a green room on a couch .", "a boy wearing headphones sits on a woman &apos;s shoulders .", "a balding man wearing a red life jacket is sitting in a small boat ."], d.source_lang_manager, d.target_lang_manager))
