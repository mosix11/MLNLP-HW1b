import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from . import utils


class Trainer():

    def __init__(self, max_epochs, lr:float=1e-5, optimizer_type="adam", use_lr_schduler=False,
                 run_on_gpu=False, gradient_clip_val=0, do_validation=True, write_summery=True):
        
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.use_lr_schduler = use_lr_schduler
        self.do_val = do_validation
        self.write_sum = write_summery
        self.cpu = utils.get_cpu_device()
        self.gpu = utils.get_gpu_device()
        
        if self.gpu == None and run_on_gpu:
            raise RuntimeError("""gpu device not found!""")
        self.run_on_gpu = run_on_gpu
        
        if self.write_sum:
            self.writer = SummaryWriter()

    def prepare_data(self, data):
        self.train_dataloader = data.get_train_dataloader()
        self.val_dataloader = data.get_val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_batch(self, batch):
        if self.run_on_gpu:
            batch = [a.to(self.gpu) for a in batch]
        return batch

    def prepare_model(self, model):
        if self.run_on_gpu:
            model.to(self.gpu)
        self.model = model

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "rmsprop":
            return torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            raise RuntimeError("Invalide optimizer type")
    
    def configure_lr_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=10)
    
    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = self.configure_optimizers()
        if self.use_lr_schduler:
            self.lr_scheduler = self.configure_lr_scheduler(self.optim)
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        # self.train_loss_hist = []
        # self.val_loss_hist = []
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
        # self.model.save_plot()
        if self.write_sum:
            self.writer.flush()

    def fit_epoch(self):
        print('#########  Entering Epoch {} #########'.format(self.epoch + 1))
        self.model.train()
        for i, batch in enumerate(self.train_dataloader):
            loss = self.model.training_step(self.prepare_batch(batch))
            if self.write_sum:
                self.writer.add_scalar('Loss/Train', loss, self.epoch*self.num_train_batches + i)
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)

                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None or not self.do_val:
            return
        self.model.eval()
        for i, batch in enumerate(self.val_dataloader):
            with torch.no_grad():
                loss, acc = self.model.validation_step(self.prepare_batch(batch))
                if self.use_lr_schduler:
                    self.lr_scheduler.step(loss)
                    print(self.lr_scheduler.get_last_lr())
                if self.write_sum:
                    self.writer.add_scalar('Loss/Val', loss, self.epoch*self.num_val_batches + i)
                    self.writer.add_scalar('Acc/Val', acc, self.epoch*self.num_val_batches + i)
            self.val_batch_idx += 1