from src import DisCoTex
from src import Trainer
from src import SentClasLSTM, SentRegLSTM, SentRegAttLSTM, HierarchicalSentRegLSTM
import torch
from pathlib import Path
import os
import functools

# import optuna
# from optuna.trial import TrialState

# import ray
from ray import train, tune
from ray.tune import CLIReporter
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler



def get_gpu_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else : return None
    
def get_cpu_device():
    return torch.device('cpu')



def define_model(config, dataset):
    if config['model_type'] == "HSR":
        model = HierarchicalSentRegLSTM(
                vocab_size=dataset.tokenizer.get_vocab_size(),
                embed_dim=config['embedding_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                bidirectional=True,
                padd_index=dataset.tokenizer.get_pad_idx(),
                dropout=config['dropout'],
                loss_fn=config['loss_fn']
            )
    elif config['model_type'] == "SR":
        model = SentRegLSTM(
                vocab_size=dataset.tokenizer.get_vocab_size(),
                embed_dim=config['embedding_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                bidirectional=True,
                padd_index=dataset.tokenizer.get_pad_idx(),
                dropout=config['dropout'],
                loss_fn=config['loss_fn']
            )
    elif config['model_type'] == "SRA":
        model = SentRegAttLSTM(
                vocab_size=dataset.tokenizer.get_vocab_size(),
                embed_dim=config['embedding_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                bidirectional=True,
                padd_index=dataset.tokenizer.get_pad_idx(),
                dropout=config['dropout'],
                loss_fn=config['loss_fn']
            )
    elif config['model_type'] == "SC":
        model = SentClasLSTM(
                vocab_size=dataset.tokenizer.get_vocab_size(),
                embed_dim=config['embedding_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                bidirectional=True,
                padd_index=dataset.tokenizer.get_pad_idx(),
                dropout=config['dropout']
            )

    return model


    

def objective(config, data_dir:Path=None, outputs_dir:Path=None):
    
    dataset = DisCoTex(root_dir=data_dir, outputs_dir=outputs_dir, batch_size=config['batch_size'], tokenizer=config['tokenizer'])
    
    model = define_model(config, dataset)
    
    trainer = Trainer(max_epochs=30, lr=config['lr'], optimizer_type=config['optimizer_type'],
                      run_on_gpu=True, do_validation=True, write_summery=False, outputs_dir=outputs_dir, ray_tuner=train)
    
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            checkpoint = torch.load(
                os.path.join(loaded_checkpoint_dir, "ckp.pt")
            )
            trainer.fit(model, dataset, checkpoint)
    else:
        trainer.fit(model, dataset)
            
        
    
    # test_loader = dataset.get_val_dataloader()
    # preds, labels = [], []
    
    # for i, batch in enumerate(test_loader):
    #     batch = [a.to(get_gpu_device()) for a in batch]
    #     _, predictions = model.predict(batch[0], batch[1])

    #     preds += predictions.tolist()
    #     labels += batch[2].tolist()
        
    # acc = model.accuracy(torch.tensor(preds), torch.tensor(labels)).cpu().item()


if __name__ == "__main__":
    data_dir = Path(os.path.abspath("./data"))
    outputs_dir = Path(os.path.abspath("./outputs/"))
    if not outputs_dir.exists():
        os.mkdir(outputs_dir)
    config = {
        'model_type': tune.choice(['SC', 'SR', 'SRA', 'HSR']),
        'tokenizer': tune.choice(['word', 'bpe']),
        'loss_fn': tune.choice(['MSE', 'MAE']),
        'embedding_size': tune.choice([64, 96, 128, 192, 265]),
        'hidden_size': tune.choice([2**i for i in range(5, 9)]),
        'num_layers': tune.choice([2, 3, 4, 5, 6]),
        'dropout': tune.choice([0.2, 0.3, 0.4, 0.5]),
        'lr': tune.loguniform(1e-5, 1e-4),
        'optimizer_type': tune.choice(['adam', 'rmsprop', 'sgd']),
        'batch_size': tune.choice([16, 32, 64]),
    }
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=30,
        grace_period=3,
        reduction_factor=2)
    
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    tuner = tune.Tuner(
        trainable=tune.with_resources(
            tune.with_parameters(objective, data_dir=data_dir, outputs_dir=outputs_dir),
            resources={"cpu": 4, "gpu": 0.25}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=500,
        ),
        
        param_space=config,
    )
    results = tuner.fit()
    
    # Extract the best trial run from the search.
    # best_trial = results.get_best_trial('loss', 'min', 'last')
    best_trial = results.get_best_result('accuracy', 'max')

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.metrics["accuracy"]))

    
# if __name__ == "__main__":
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=50)

#     pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
#     complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

#     print("Study statistics: ")
#     print("  Number of finished trials: ", len(study.trials))
#     print("  Number of pruned trials: ", len(pruned_trials))
#     print("  Number of complete trials: ", len(complete_trials))

#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: ", trial.value)

#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))
        
        

