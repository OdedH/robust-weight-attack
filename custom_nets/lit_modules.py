from abc import ABC
import lightning as L
import torch
from torchmetrics import functional as FM
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class BasicLitModule(L.LightningModule, ABC):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss
        # self.optimizer = optimizer
        # self.scheduler = scheduler
        self.save_hyperparameters()

    def configure_optimizers(self):
        # return self.optimizer
        return torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss)
        preds = torch.argmax(logits, dim=-1)
        acc = FM.accuracy(preds, y, num_classes=43, task="multiclass")
        metrics = {'train_acc': acc, 'train_loss': loss}
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log('test_loss', loss)
        preds = torch.argmax(logits, dim=-1)
        acc = FM.accuracy(preds, y, num_classes=43, task="multiclass")
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)
        return loss

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="train_loss", patience=10, mode="min")
        checkpoint = ModelCheckpoint(
            monitor='train_loss',
            dirpath='./checkpoints/',
            filename='Base-Regular-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
        )
        return [early_stop, checkpoint]
