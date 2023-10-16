import glob
import threading
import time

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, RLock

label_name = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]


# 数据处理对象
class DataCope:
    def __init__(self):
        super().__init__()
        # 标签映射名
        self.label_name = []
        self.label_dict = {}
        for num, name in enumerate(label_name):
            self.label_dict[name] = num
        # 数据处理对象
        self.train_transform = None
        self.test_transform = None
        # print(self.label_name)

    def data_transform(self):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomGrayscale(0.1),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor()
        ])
        return self.train_transform, self.test_transform


def default_loader(path: str):
    return Image.open(path).convert("RGB")


# 数据加载类
class MyDataset(Dataset):
    def __init__(self, im_list: list, transform=None, loader=default_loader):
        super().__init__()
        imgs = []
        for im_item in im_list:
            im_label_name = im_item.split("\\")[-2]
            imgs.append([im_item, cope.label_dict[im_label_name]])
        self.imgs = imgs
        self.transform = transform
        self.list = im_list
        self.loader = loader

    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)

        if self.transform is not None:
            im_data = self.transform(im_data)
        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


class MyThread(threading.Thread):
    def __init__(self, func, args):
        super().__init__()
        self.func = func
        self.args = args
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


# 数据处理类
cope = DataCope()
# 互斥锁初始化
q = RLock()
# 得到列表任务
im_train_list = glob.glob(r"./cifar-10-batches-py/train/*/*.png")
im_test_list = glob.glob(r"./cifar-10-batches-py/test/*/*.png")

# 初始处理任务
train_dataset = MyDataset(im_list=im_train_list, transform=cope.data_transform()[0])
test_dataset = MyDataset(im_list=im_test_list, transform=cope.data_transform()[1])

# 数据加载任务
train_data_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
print(f"num_of_train={len(train_dataset)}")
print(f"num_of_test={len(test_dataset)}")

# task3 = []
# train_data_loader = None
# test_data_loader = None
# all_data_loader = [train_data_loader, test_data_loader]
# for i in range(2):
#     t3 = MyThread(DataLoader, (all_dataset[i], 64, True))
#     t3.start()
#     task3.append(t3)
# for idx, t3 in enumerate(task3):
#     t3.join()
#     all_data_loader[idx] = t3.get_result()
