
import torch
from tqdm import tqdm
from torch import optim

from torch.nn import CrossEntropyLoss, Softmax

from torchmetrics import Accuracy


class Trainer:
    def __init__(self, model, optimizer, log_function, accelerator, *args, **kwargs):
        self.accelerator = accelerator
        self.model = model
        self.model.to(self.accelerator)
        self.accuracy = Accuracy(task='multiclass', num_classes=10)
        self.accuracy.to(self.accelerator)
        
        self.optimizer = optimizer
        self.log_function = log_function # TODO: add logging for desired metrics
        
        # a list for saving the best models
        self.best_model = None
        
    def train(self, train_loader, val_loader, epochs=10):
        #peform training iterations
        
        for epoch in range(epochs):
            self.train_epoch(train_loader, epoch)
            self.validate_epoch(val_loader, epoch)
            
    @torch.enable_grad()
    def train_epoch(self, train_loader, current_epoch):
        #define the loss fn and optimizer
        batch_losses = []

        #reset iterator
        dataiter = iter(train_loader)
        
        # use tqdm progress bar as wrapper for the data loader
        with tqdm(dataiter, bar_format='{l_bar}{bar:10}{r_bar}') as pbar:
            # iterate through epoch by iterating through the data loader
            for batch_data, batch_lbl in pbar:
                
                # set progress bar description
                pbar.set_description(f"Training Epoch {current_epoch}")
                        
                #reset gradients
                self.optimizer.zero_grad()

                out = self.model(batch_data)
                
                #calculate the loss
                loss = self.train_loss_per_step(out, batch_lbl)
                self.log_function({"train_loss": loss.item()})
                
                batch_losses.append(loss.item())
                loss.backward()

                self.optimizer.step()
                
                # set progress bar stats
                pbar.set_postfix(loss=loss.item())

        return batch_losses
            
    def on_val_epoch_end(self, outputs, losses):        
        # save the model if it is the best one so far
        if self.best_model is None or torch.mean(torch.tensor(losses)) < torch.mean(torch.tensor(self.best_model["losses"])):
            self.best_model = {
                "model_state": self.model.state_dict(),
                "losses": losses,
            }
    
    def train_loss_per_step(self, out, batch_lbl):
        # define the loss fn
        batch_lbl = batch_lbl.to(self.accelerator)
        loss_fn = CrossEntropyLoss()
        return loss_fn(out, batch_lbl)
        
 
    @torch.no_grad()
    def validate_epoch(self, val_loader, current_epoch):
        # save model output and loss for later metrics
        losses = []
        outputs = []
        labels = []
        
        # reset iterator
        dataiter = iter(val_loader)
        
        # use tqdm progress bar as wrapper for the data loader
        with tqdm(dataiter, bar_format='{l_bar}{bar:10}{r_bar}') as pbar:
            # iterate through epoch by iterating through the data loader
            for batch_data,batch_lbl in pbar:
                # set progress bar description
                pbar.set_description(f"Validation Epoch {current_epoch}")
                
                out = self.model(batch_data)
                loss = self.validation_loss_per_step(out, batch_lbl)
                
                outputs.append(out)
                labels.append(batch_lbl)
                losses.append(loss.item())
                
                self.log_function({"val_loss": loss.item()})
                losses.append(loss)
                
                # set progress bar stats
                pbar.set_postfix(loss=loss)
                
        self.on_val_epoch_end(outputs, losses)
        self.validation_loss_per_epoch(outputs, labels)
        
    
    def validation_loss_per_step(self, out, batch_lbl):
        # depending on what shall be logged during validation, this function can be overwritten
        batch_lbl = batch_lbl.to(self.accelerator)
        loss_fn = CrossEntropyLoss()
        return loss_fn(out, batch_lbl)
    
    def validation_loss_per_epoch(self, outputs, labels):
        labels = torch.cat(labels)
        labels = labels.to(self.accelerator)
        outputs = torch.cat(outputs)
        outputs = Softmax(dim=1)(outputs)
        
        acc = self.accuracy(outputs, labels).item()
        
        self.log_function({"val_acc": acc})

    
def configure_optimizer(model, learning_rate):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer
