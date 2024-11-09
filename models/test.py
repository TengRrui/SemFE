import torch
from model import MatchingNet

def test_matching_net():
    # 构造测试输入数据
    samples0 = torch.randn(1, 3, 320, 320)  # 假设输入图像大小为 256x256，通道数为 3
    samples1 = torch.randn(1, 3, 320, 320)
    gt_matrix = None  # 假设未提供真实的匹配矩阵
    # 创建匹配网络模型
    matching_net = MatchingNet().to('cuda:0')
    samples0 = samples0.to('cuda:0')
    samples1 = samples1.to('cuda:0')

    # 执行前向传播
    output = matching_net(samples0, samples1, gt_matrix)

    # 打印输出信息
    print("Output keys:", output.keys())
    print("Shape of cm_matrix:", output['cm_matrix'].shape)
    print("Shape of matches:", output['matches'].shape)
    print("Shape of mkpts0:", output['mkpts0'].shape)
    print("Shape of mkpts1:", output['mkpts1'].shape)
    print("Shape of mdesc0:", output['mdesc0'].shape)
    print("Shape of mdesc1:", output['mdesc1'].shape)

    # 输出匹配结果
    print("Matching results:")
    print("Number of matches:", output['matches'].shape[0])
    for i, match in enumerate(output['matches']):
        print(f"Match {i + 1}: {match}")

# 执行测试函数
test_matching_net()
