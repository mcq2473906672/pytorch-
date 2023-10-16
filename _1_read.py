import glob
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

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


class ShowPhtot:
    def __init__(self, save_path: str, read_path: str):
        super().__init__()
        self.save_path = save_path
        self.quality = self.save_path.split("/")[-1]
        self.read_path = read_path
        self.quality_list = []

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            l_dict = pickle.load(fo, encoding='bytes')
        return l_dict

    def run(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.quality_list = glob.glob(self.read_path)
        for i in self.quality_list:
            l_dict = self.unpickle(i)
            for im_idx, im_data in enumerate(l_dict[b'data']):
                # print(im_idx)
                # print(im_data.shape)
                im_label = l_dict[b'labels'][im_idx]
                im_name = l_dict[b'filenames'][im_idx]
                # print(im_label)
                # print(im_name)

                im_label_name = label_name[im_label]
                im_data = np.reshape(im_data, [3, 32, 32])
                im_data = np.transpose(im_data, (1, 2, 0))
                im_data = cv2.resize(im_data, (320, 320))
                # cv2.imshow("im_data",im_data)
                # cv2.waitKey()

                if not os.path.exists(f"{self.save_path}/{im_label_name}"):
                    os.mkdir(f"{self.save_path}/{im_label_name}")
                cv2.imwrite(f"{self.save_path}/{im_label_name}/{im_name.decode('utf-8')}", im_data)
        return None


if __name__ == '__main__':
    read_test = ShowPhtot("./cifar-10-batches-py/test", "./cifar-10-batches-py/test_batch")
    read_train = ShowPhtot("./cifar-10-batches-py/train", "./cifar-10-batches-py/data_batch_*")

    # 单线程
    # start = time.time()
    # read_train.run()
    # read_test.run()
    # print(time.time() - start)  # 134

    # 线程池
    start = time.time()
    pool = ThreadPoolExecutor(2)
    f_list = []
    future = pool.submit(read_train.run)
    f_list.append(future)
    future = pool.submit(read_test.run)
    f_list.append(future)
    pool.shutdown()  # 等待任务完成
    for f in f_list:
        print(f"{f.result}")
    print(time.time() - start)  # 28s
