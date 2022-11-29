from KITTI_depth_dataset import *
import torchvision


data_root='/Users/spcolin/datasets/'
gt_root='/Users/spcolin/datasets/data_depth_annotated/train/'
file_path='eigen_train_files_with_gt_small.txt'

target_height=352
target_width=1120



dataset=KITTI_Depth(data_path=data_root,gt_path=gt_root,file_path=file_path)

sample=dataset[3]

rgb=sample['image']
depth=sample['depth']

rgb=torchvision.transforms.ToPILImage()(rgb)
# depth=torchvision.transforms.ToPILImage()(depth)

rgb.show()
depth.show()

