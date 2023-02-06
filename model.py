
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from utils import *
from data_setup import classes


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        # labels = labels.float().unsqueeze(1)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        # print('training loss and acc:', loss, acc)
        return loss, acc
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        # labels = labels.float().unsqueeze(1)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        # print('Validation loss and acc:', loss, acc)
        return {'val_loss':loss.detach(), 'val_acc':acc}

    def validation_end_epoch(self, results):
        batch_loss = [x['val_loss'] for x in results]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x['val_acc'] for x in results]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss':epoch_loss.item(), 'val_acc':epoch_acc.item()}

    # def epoch_end(self, epoch, outputs):
    #     print(f"Epoch {epoch+1}: train_loss: {outputs['train_loss']}, val_loss: {outputs['val_loss']}, val_acc: {outputs['val_acc']}")

    def epoch_end(self, epoch, result):
        print(f"Epoch {epoch+1}: train_loss: {result['train_losses']:.4f}, train_acc: {result['train_acc']:.4f}, \
        val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f} ")


class Efficient_b2_model(ImageClassificationBase):
    def __init__(self, num_classes=len(classes), pretrained=True):
        super().__init__()
        if pretrained:
            if torchvision.__version__ >= '0.13.0':
                self.network = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT)
            
            else:
                # 1. Get the base mdoel with pretrained weights and send to target device
                self.network = torchvision.models.efficientnet_b2(pretrained=True)

            for param in self.network.parameters():
                param.requires_grad =False
            
            self.network.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                                 nn.Linear(in_features=1408, out_features=num_classes, bias=True)
                                                 )
        else:
            self.network = torchvision.models.efficientnet_b2()
    

    def forward(self, x):
        x = self.network(x)
        return x
