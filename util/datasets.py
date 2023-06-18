# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# UM-MAE: https://github.com/implus/UM-MAE
# --------------------------------------------------------

import os
import PIL
import torchvision.transforms
from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ImageListFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ann_file=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.nb_classes = 1000

        assert ann_file is not None
        print('load info from', ann_file)

        self.samples = []
        ann = open(ann_file)
        for elem in ann.readlines():
            cut = elem.split(' ')
            path_current = os.path.join(root, cut[0])
            target_current = int(cut[1])
            self.samples.append((path_current, target_current))
        ann.close()

        print('load finish')


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    # TODO modify your own dataset here
    folder = os.path.join(args.data_path, 'train' if is_train else 'val')
    ann_file = os.path.join(args.data_path, 'train.txt' if is_train else 'val.txt')
    dataset = ImageListFolder(folder, transform=transform, ann_file=ann_file)

    print(dataset)

    return dataset

from util.my_dataset import *
def read_split_data(root: str, val_rate: float = 0.2):
    import random
    import json
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)
    return train_images_path, train_images_label, val_images_path, val_images_label
def build_dataset_mine(args):

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset_name.startswith("PlantDoc"):
        root_train = os.path.join(args.data_path, 'train')
        train_dataset = datasets.ImageFolder(root_train, transform=transform_train)
        root_val = os.path.join(args.data_path, 'val')
        val_dataset = datasets.ImageFolder(root_val, transform=transform_train)

    if args.dataset_name.startswith("plantv"):
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
        # 实例化训练数据集
        train_dataset = MyDataSet(images_path=train_images_path,
                                  images_class=train_images_label,
                                  transform=transform_train)

        # 实例化验证数据集
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=transform_val)
    if args.dataset_name.startswith("IP102"):
        train_dataset = IP102(txt_path=os.path.join(args.data_path, "train_new.txt"), transform=transform_train)
        val_dataset = IP102(txt_path=os.path.join(args.data_path, "test_new.txt"), transform=transform_val)

    if args.dataset_name.startswith("deepweeds"):
        train_dataset = DeepWeeds(csv_path=os.path.join(args.data_path, "train.csv"), transform=transform_train)
        val_dataset = DeepWeeds(csv_path=os.path.join(args.data_path, "test.csv"), transform=transform_val)
    return train_dataset,val_dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
