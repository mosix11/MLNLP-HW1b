import torch
import random
from .models import SentRegRNN
from .trainer import Trainer
from .utils import get_gpu_device, get_cpu_device

def straified_baseline(dataset):
    train_dl = dataset.get_train_dataloader()
    val_dl = dataset.get_val_dataloader()
    test_dl = dataset.get_test_dataloader()
    
    label_freqs = {0: 0, 1: 0, 2: 0, 3: 0}
    for batch in train_dl:
        paded, lens, labels = batch
        for l in labels:
            label_freqs[int(l)] = label_freqs[int(l)] + 1
    for batch in val_dl:
        paded, lens, labels = batch
        for l in labels:
            label_freqs[int(l)] = label_freqs[int(l)] + 1
            
    keys = list(label_freqs.keys())
    counts = list(label_freqs.values())
    total_samples = sum(counts)
    probabilities = [count/total_samples for count in counts]
    
    accuracy = []
    for batch in test_dl:
        paded, lens, labels = batch
        for target in labels:
            prediction = random.choices(keys, weights=probabilities, k=1)
            accuracy.append(float(prediction[0] == int(target.cpu().item())))
    print("Accuracy on test set is: ", sum(accuracy) / len(accuracy))
    # print(label_freqs)
    # print(probabilities)
    

def simple_rnn_baseline(dataset):
    model = SentRegRNN(
            vocab_size=dataset.tokenizer.get_vocab_size(),
            embed_dim=128,
            hidden_size=64,
            num_layers=1,
            padd_index=dataset.tokenizer.get_pad_idx(),
        )
    trainer = Trainer(max_epochs=500, lr=1e-5, optimizer_type="rmsprop", run_on_gpu=True, use_lr_schduler=True)
    trainer.fit(model, dataset)
    
    test_loader = dataset.get_test_dataloader()
    preds, labels = [], []
    for i, batch in enumerate(test_loader):
        if get_gpu_device() != None:
            batch = [a.to(get_gpu_device()) for a in batch]
        _, predictions = model.predict(batch[0], batch[1])

        preds += predictions.tolist()
        labels += batch[2].tolist()
        
    print("Accuracy on test set is: ", model.accuracy(torch.tensor(preds), torch.tensor(labels)))