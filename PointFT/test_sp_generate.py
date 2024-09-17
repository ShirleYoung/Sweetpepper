import pickle
import torch
import torch.utils.data as data
import os
import numpy as np
import open3d as o3d
import json
import cv2
from torch.utils.data import Dataset, DataLoader

class IGGFruit(Dataset):
    def __init__(self, data_source=None, split='train', return_pcd=True, return_rgbd=False, num_points=2048):
        assert return_pcd or return_rgbd, "return_pcd and return_rgbd are set to False. Set at least one to True"
        self.data_source = data_source
        self.split = split
        self.return_pcd = return_pcd
        self.return_rgbd = return_rgbd
        self.num_points = num_points
        self.fruit_list = self.get_file_paths()
        self.preprocessed_data = self.preprocess_data()

    def get_file_paths(self):
        fruit_list = {}
        split_dir = os.path.join(self.data_source, self.split)
        for fid in os.listdir(split_dir):
            fruit_list[fid] = {
                'path': os.path.join(split_dir, fid),
            }
        return fruit_list

    def preprocess_data(self):
        preprocessed_data = {}
        submission_dir = '../pointft_dataset'
        
        for fid in self.fruit_list.keys():
            # 根据split来构造points的路径
            ply_path = os.path.join(submission_dir, self.split, f"{fid}.ply")
            
            # 读取points
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)
            
            # 如果不是测试集，构造gt_pcd的路径并读取gt_points
            gt_points = None
            if self.split != 'test':
                gt_pcd_path = os.path.join(self.data_source, self.split, fid, 'gt/pcd/fruit.ply')
                gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
                gt_points = np.asarray(gt_pcd.points)
            
            # 下采样points
            downsampled_pcd = self.fps_downsample(pcd, self.num_points)
            downsampled_points = np.asarray(downsampled_pcd.points)
            
            # 存储预处理后的数据
            preprocessed_data[fid] = {
                "points": downsampled_points,
                "gt_points": gt_points if gt_points is not None else np.empty((0, 3)),  # 确保gt_points始终是一个数组
                "filename": ply_path
            }
        
        return preprocessed_data


    def get_gt(self, fid):
        return o3d.io.read_point_cloud(os.path.join(self.fruit_list[fid]['path'], 'gt/pcd/fruit.ply'))

    def get_rgbd(self, fid):
        fid_root = self.fruit_list[fid]['path']
        intrinsic_path = os.path.join(fid_root, 'input/intrinsic.json')
        intrinsic = self.load_K(intrinsic_path)
        
        rgbd_data = {
            'intrinsic': intrinsic,
            'pcd': o3d.geometry.PointCloud(),
            'frames': {}
        }

        frames = os.listdir(os.path.join(fid_root, 'input/masks/'))
        for frameid in frames:
            pose_path = os.path.join(fid_root, 'input/poses/', frameid.replace('png', 'txt'))
            pose = np.loadtxt(pose_path)
            
            rgb_path = os.path.join(fid_root, 'input/color/', frameid)
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

            depth_path = os.path.join(fid_root, 'input/depth/', frameid.replace('png', 'npy'))
            depth = np.load(depth_path)

            mask_path = os.path.join(fid_root, 'input/masks/', frameid)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            frame_key = frameid.replace('png', '')

            if self.return_pcd:
                frame_pcd = self.rgbd_to_pcd(rgb, depth, mask, pose, intrinsic)
                rgbd_data['pcd'] += frame_pcd

            rgbd_data['frames'][frame_key] = {
                'rgb': rgb,
                'depth': depth,
                'mask': mask,
                'pose': pose
            }

        return rgbd_data

    @staticmethod
    def load_K(path):
        with open(path, 'r') as f:
            data = json.load(f)['intrinsic_matrix']
        k = np.reshape(data, (3, 3), order='F')
        return k

    @staticmethod
    def rgbd_to_pcd(rgb, depth, mask, pose, K):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth * mask),
            depth_scale=1,
            depth_trunc=1.0,
            convert_rgb_to_intensity=False
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            height=rgb.shape[0],
            width=rgb.shape[1],
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2]
        )

        extrinsic = np.linalg.inv(pose)
        frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
        return frame_pcd

    def __len__(self):
        return len(self.fruit_list)
    
    @staticmethod
    def fps_downsample(pcd, num_points):
        print("downsample begins.")
        if len(pcd.points) <= num_points:
            return pcd

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        points = torch.tensor(np.asarray(pcd.points), device=device, dtype=torch.float32)
        
        N, _ = points.shape
        centroids = torch.zeros(num_points, dtype=torch.long, device=device)
        distances = torch.ones(N, device=device) * 1e10
        farthest = torch.randint(0, N, (1,), device=device).item()

        for i in range(num_points):
            centroids[i] = farthest
            centroid = points[farthest]
            dist = torch.sum((points - centroid) ** 2, dim=1)
            mask = dist < distances
            distances[mask] = dist[mask]
            farthest = torch.argmax(distances).item()

        downsampled_points = points[centroids].cpu().numpy()
        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
        return downsampled_pcd

    def __getitem__(self, idx):
        keys = list(self.fruit_list.keys())
        fid = keys[idx]
        item = self.preprocessed_data[fid]
        points = item["points"]
        gt_points = item["gt_points"]
        label = 0
        filename = item["filename"]
        
        if self.split != 'test':
            return label, points, gt_points, filename
        else:
            return label, points, points, filename

    @staticmethod
    def collate_fn(batch):
        labels = torch.tensor([item[0] for item in batch], dtype=torch.int64)
        points = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        if isinstance(batch[0], tuple) and len(batch[0]) == 3:
            gt_points = torch.tensor([item[2] for item in batch], dtype=torch.float32)
            return labels, points, gt_points
        else:
            return labels, points, points


import torch.optim as optim
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse

def load_or_create_dataset(split):
    dataset_save_path = os.path.join(args.data_source, f'{split}_dataset_pft.pkl')

    if os.path.exists(dataset_save_path):
        # 如果存在预处理后的数据集，则从文件中加载
        with open(dataset_save_path, 'rb') as f:
            # 确保 IGGFruit 类已经被定义
            dataset = pickle.load(f)
        print(f"从 {dataset_save_path} 加载 {split} 数据集")
    else:
        # 创建数据集并保存到文件
        dataset = IGGFruit(
            data_source=args.data_source,
            split=split,
            return_pcd=True,
            return_rgbd=False
        )
        print(f"{split} 数据集大小: {len(dataset)}")
        
        # 将数据集保存到文件
        with open(dataset_save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"将 {split} 数据集保存到 {dataset_save_path}")
    
    # 新增代码以输出数据集中样本的数据结构
    if len(dataset) > 0:
        print(f"{split} 样本数据结构:")
        sample = dataset[0]
        if isinstance(sample, tuple):
            print(f"  label: {type(sample[0])}")
            print(f"  points: {type(sample[1])}, shape: {sample[1].shape}")
            if len(sample) > 2:
                print(f"  gt_points: {type(sample[2])}, shape: {sample[2].shape}")
    
    return dataset

def load_model(model_path, model_module, args):
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    ckpt = torch.load(model_path)
    net.module.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    return net

def generate_predictions(model, dataloader, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for data in dataloader:
            label, inputs, gt, filenames = data  # 假设 __getitem__ 现在返回 filename
            gt = gt.float().cuda()
            inputs = inputs.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()
            print(f"Batch x: inputs shape = {inputs.shape}")
            
            out, _, _ = model(inputs, gt)
            print(f"Batch x: out shape = {out.shape}")
            if out is None:
                print(f"Model output is None for batch x!")
                continue  # 跳过这个batch

            # 保存每个样本的预测结果
            for i, prediction in enumerate(out):
                prediction_np = prediction.cpu().numpy().reshape(-1, 3)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(prediction_np)
                
                # 构造保存路径
                original_filename = filenames[i]
                base_filename = os.path.basename(original_filename)
                save_path = os.path.join(save_dir, base_filename)
                
                o3d.io.write_point_cloud(save_path, pcd)
                print(f"Saved point cloud to {save_path}")

def save_predictions(predictions, original_files, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, (prediction, original_file) in enumerate(zip(predictions, original_files)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(prediction)
        
        # 使用原始文件名来构造保存路径
        original_filename = os.path.basename(original_file)
        save_path = os.path.join(save_dir, original_filename)
        
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Saved point cloud to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = 'cfgs/PointFT.yaml'
    args = munch.munchify(yaml.safe_load(open(config_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 初始化数据集
    if args.dataset == 'iggfruit':
        dataset_test = load_or_create_dataset(split='test')
    else:
        raise ValueError('Dataset does not exist')

    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))

    # 获取原始残缺点云文件的路径
    original_files = [dataset_test.preprocessed_data[fid]["filename"] for fid in dataset_test.fruit_list.keys()]

    # 加载模型模块
    model_module = importlib.import_module('.%s' % args.model_name, 'models')

    # 加载训练好的模型
    model_path = './log/Pointft_cd_debug_pcn/network.pth'
    model = load_model(model_path, model_module, args)

    # 生成预测
    save_dir = '../pointft_result'
    generate_predictions(model, dataloader_test, save_dir)
