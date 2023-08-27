import torch
import torchvision
import pytorch_lightning as pl
from torch.nn import functional as F
from lit_modules import BasicLitModule
from argparse import ArgumentParser
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader
from models import quan_resnet
from pytorch_lightning.loggers import WandbLogger
import lightning as L
import torchvision.transforms as transforms


def cli_main():
    pl.seed_everything(42)
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    # ------------
    # data
    # ------------
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])
    dataset_train = GTSRB(root="../data/GTSRB", split='train', transform=transform)
    dataset_test = GTSRB(root="../data/GTSRB", split='test',transform=transform)
    train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False)
    # ------------
    # model
    # ------------
    loss = F.cross_entropy
    class_num = 43
    input_size = 32
    bit_length = 8
    resnet20_quant = torch.nn.DataParallel(quan_resnet.resnet20_quan_mid_custom(class_num, bit_length))
    # ------------
    # logging
    # ------------
    wandb_logger = WandbLogger(name='Base-Network', project='robust-weight-attack')
    # ------------
    # training
    # ------------
    trainer = L.Trainer(
        accelerator="gpu", devices=[0],
        max_epochs=300,
        logger=wandb_logger
    )
    trainer.fit(
        BasicLitModule(resnet20_quant, loss),
        train_dataloaders=train_loader,
    )
    # ------------
    # testing
    # ------------
    trainer.test(
        BasicLitModule(resnet20_quant, loss),
        dataloaders=test_loader,)


if __name__ == '__main__':
    cli_main()


