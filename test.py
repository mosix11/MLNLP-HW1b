from src import DisCoTex
from src import Trainer
from src import SentClasLSTM


ds = DisCoTex(tokenizer='bpe')

dl = ds.get_train_dataloader()
for batch in dl:
    paded, lens, labels = batch

    # print(type(lens), lens.shape)
    # print(paded.shape)
    # print(type(labels), labels.shape)
    print(ds.tokenizer.decode(paded[0].tolist()))
    exit()
    