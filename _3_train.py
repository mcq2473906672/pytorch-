import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os.path
import torch.nn as nn
from _2_load import train_dataset, test_dataset, train_data_loader, test_data_loader
from pre_resnet import pytorch_resnet34
from threading import RLock


class Paramater:
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch_num = 20
        self.lr = 0.01
        self.net = pytorch_resnet34().to(self.device)
        self.log_path = "pytorch_resnet34_log"
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        self.writer = SummaryWriter(log_dir=f"./{self.log_path}")
        self.model_path = "pytorch_resnet34_model"
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.loss_fc = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)

    def run(self):
        for epoch in range(self.epoch_num):
            print(f"epoch is {epoch + 1}")
            self.net.train()
            train_loss = 0
            for idx, data in enumerate(train_data_loader):
                imgs, targets = data
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.net(imgs)
                train_loss = self.loss_fc(outputs, targets)

                print(f"第{idx}次，训练集上LOSS：{train_loss}")
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
            self.writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=epoch)

            self.net.eval()
            total_loss_test = 0
            total_accuracy_test = 0
            with torch.no_grad():
                for idx, data in enumerate(test_data_loader):
                    imgs, targets = data
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.net(imgs)
                    test_loss = self.loss_fc(outputs, targets)
                    total_loss_test += test_loss.item()
                    accuracy = (outputs.argmax(1) == targets).sum()
                    total_accuracy_test += accuracy
                print(f"测试集上loss：{total_loss_test}")
                print(f"测试集上准确率：{total_accuracy_test / len(test_data_loader)}")
                self.writer.add_scalar(tag="test_loss", scalar_value=total_loss_test, global_step=epoch)
                self.writer.add_scalar(tag="test_acc", scalar_value=total_accuracy_test / len(test_dataset),
                                       global_step=epoch)

            torch.save(self.net.state_dict(), f"{self.model_path}/{epoch + 1}")
            # self.scheduler.step()
        self.writer.close()


train = Paramater()
train.run()
# q = RLock()
