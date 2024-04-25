from src import DisCoTex
from src import Trainer
from src import SentClasLSTM, SentRegLSTM
import torch
from pathlib import Path

import optuna
from optuna.trial import TrialState

dataset = DisCoTex(root=Path('./data'), batch_size=32, tokenizer="bpe")


def get_gpu_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else : return None
    
def get_cpu_device():
    return torch.device('cpu')

def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    embedding_size = trial.suggest_int('embedding_size', 50, 300)
    hidden_size = trial.suggest_categorical('hidden_size', [2**i for i in range(5, 10)])
    num_layers = trial.suggest_int('num_layers', 1, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    model = SentRegLSTM(
            vocab_size=dataset.tokenizer.get_vocab_size(),
            embed_dim=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            padd_index=dataset.tokenizer.get_pad_idx(),
            dropout=dropout_rate
        )

    return model


    

def objective(trial):
    
    model = define_model(trial)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optim_type = trial.suggest_categorical('optimizer_type', ['adam', 'adamw', 'sgd', 'rmsprop'])
    trainer = Trainer(max_epochs=20, lr=lr, optimizer_type=optim_type, run_on_gpu=True, do_validation=False, write_summery=False)
    trainer.fit(model, dataset)
    
    test_loader = dataset.get_val_dataloader()
    preds, labels = [], []
    for i, batch in enumerate(test_loader):
        
        batch = [a.to(get_gpu_device()) for a in batch]
        _, predictions = model.predict(batch[0], batch[1])

        preds += predictions.tolist()
        labels += batch[2].tolist()
        
    acc = model.accuracy(torch.tensor(preds), torch.tensor(labels)).cpu().item()
    return acc



    
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
        

