from datasets.rotate_rgbd_dataset import build_Rotate_RGBD
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

train_data_file = "D:/study/data/rgbd/train.txt"
test_data_file = "D:/study/data/rgbd/val.txt"
train_dataset, test_dataset = build_Rotate_RGBD(train_data_file, test_data_file, size=(320, 320), stride=8)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

for batch_idx, sample in enumerate(train_loader):
    refer_images = sample["refer"]  # 光学图像
    query_images = sample["query"]  # SAR图像
    labels = sample["gt_matrix"]    # 标签

    # 这里可以添加模型的训练代码，用于测试数据加载和处理流程是否正确
    # 比如输出样本的形状等信息
    print("Batch:", batch_idx)
    print("Refer Images Shape:", refer_images.shape)
    print("Query Images Shape:", query_images.shape)
    print("Labels Shape:", labels.shape)