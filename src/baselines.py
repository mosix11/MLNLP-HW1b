import torch
import random

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
    print(sum(accuracy) / len(accuracy))
    # print(label_freqs)
    # print(probabilities)