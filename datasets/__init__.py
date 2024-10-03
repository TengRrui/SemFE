from datasets.depth_dataset import build_depth
from datasets.sar_opt_dataset import build_so
from datasets.nirscene1_dataset import build_nir
from datasets.rgbd_dataset import build_rgbd
from datasets.SE_sar_opt_dataset import build_SE
from datasets.space_sar_opt_dataset import build_SP
from datasets.rotate_space_sar_opt_dataset import build_Rotate_SpaceNet
from datasets.rotate_rgbd_dataset import build_Rotate_RGBD

def build_dataset(args):
    if args.data_name == 'so':
        train_data_file="/home/ly/Documents/zkj/stage1/train/train.txt"
        test_data_file="/home/ly/Documents/zkj/stage1/train/test.txt"
        return build_so(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'nir':
        # train_data_file="/home/ly/Documents/zkj/nirscene1/train.txt"
        # test_data_file="/home/ly/Documents/zkj/nirscene1/train.txt"
        test_data_file="/four_disk/image_patch_dataset/rgbd/normalize_train.txt"
        test_data_file="/four_disk/image_patch_dataset/rgbd/normalize_test.txt"
        return build_nir(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'rgbd':
        train_data_file="/home/ly/Documents/zkj/rgbd/train.txt"
        test_data_file="/home/ly/Documents/zkj/rgbd/val.txt"
        return build_Rotate_RGBD(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'SE':
        train_data_file="/home/ly/Documents/zkj/ROIs/se_train.txt"
        test_data_file="/home/ly/Documents/zkj/ROIs/se_train.txt"
        return build_SE(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'SP':
        train_data_file="/home/ly/data2/SpaceNet-6/spacenet/se_train.txt"
        test_data_file="/home/ly/data2/SpaceNet-6/spacenet/se_test.txt"
        return build_SP(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'RSP':
        train_data_file="/home/ly/Documents/zkj/dataset/spacenet/se_train.txt"
        test_data_file="/home/ly/Documents/zkj/dataset/spacenet/se_test.txt"
        return build_Rotate_SpaceNet(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
