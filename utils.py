import torch
import matplotlib as plt
from torch.utils.data import DataLoader, TensorDataset

def show_example(img,label):
  print('Label: ', classes[label], '('+str(label)+')')
  plt.imshow(img.permute(1, 2, 0))


def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


def accuracy(out,labels):
    _, preds = torch.max(out,dim=1)
    total = torch.sum(preds == labels).item()/len(preds)
    return torch.tensor(total)

@torch.inference_mode()
def evaluation(model,val_loader):
    model.eval()
    results = [model.validation_step(batch) for batch in val_loader]
    outputs = model.validation_end_epoch(results)
    return outputs


def to_device(data, device):
    if isinstance(data, (tuple, list)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader(DataLoader):
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for x in self.dl:
            yield to_device(x, self.device)
        
    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        train_acc = []
        lrs = []
        for batch in train_loader:
            loss, acc = model.training_step(batch)
            train_losses.append(loss)
            train_acc.append(acc)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluation(model, val_loader)
        result['train_losses'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = torch.stack(train_acc).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def plot_accuracies(history):
    plt.plot([x['val_acc'] for x in history], '-rx')
    plt.plot([x['train_acc'] for x in history[1:]], '-bx')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Validation', 'Training'])
    plt.title('Accuracy vs. No. of epochs');

def plot_losses(history):
    plt.plot([x['val_loss'] for x in history], '-rx')
    plt.plot([x['train_losses'] for x in history[1:]], '-bx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Validation', 'Training'])
    plt.title('Loss vs. No. of epochs');


def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');
