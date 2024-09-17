import pickle
import torch.utils.data as data
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d
import json
import cv2
from pytorch_lightning import LightningDataModule
import torch

class IGGFruit(Dataset):
    def __init__(self, data_source=None, split='train', return_pcd=True, return_rgbd=True):
        assert return_pcd or return_rgbd, "return_pcd and return_rgbd are set to False. Set at least one to True"
        self.data_source = data_source
        self.split = split
        self.return_pcd = return_pcd
        self.return_rgbd = return_rgbd
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
        for fid in self.fruit_list.keys():
            gt_pcd = self.get_gt(fid) if self.split != 'test' else None
            input_data = self.get_rgbd(fid)

            points = np.asarray(input_data['pcd'].points)
            colors = np.asarray(input_data['pcd'].colors)
            gt_points = np.asarray(gt_pcd.points) if gt_pcd else None
            gt_normals = np.asarray(gt_pcd.normals) if gt_pcd else None

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            downsampled_pcd = self.fps_downsample(pcd, 50000)
            downsampled_points = np.asarray(downsampled_pcd.points)
            downsampled_colors = np.asarray(downsampled_pcd.colors)

            preprocessed_data[fid] = {
                "points": downsampled_points,
                "colors": downsampled_colors,
                "extra": {
                    "gt_points": gt_points if gt_pcd else [],
                    "gt_normals": gt_normals if gt_pcd else []
                },
                "filename": self.fruit_list[fid]['path']
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
        colors = torch.tensor(np.asarray(pcd.colors), device=device, dtype=torch.float32)
        
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
        downsampled_colors = colors[centroids].cpu().numpy()
        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
        downsampled_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)
        return downsampled_pcd

    def __getitem__(self, idx):
        keys = list(self.fruit_list.keys())
        fid = keys[idx]
        item = self.preprocessed_data[fid]
        print("Shape of points:", item["points"].shape)
        if self.split != 'test':
            print("Shape of gt_points:", item["extra"]["gt_points"].shape)
        return item

    @staticmethod
    def collate_fn(batch):
        collated_batch = {
            "points": [item["points"] for item in batch],
            "colors": [item["colors"] for item in batch],
            "extra": {
                "gt_points": [item["extra"]["gt_points"] for item in batch if "gt_points" in item["extra"]],
                "gt_normals": [item["extra"]["gt_normals"] for item in batch if "gt_normals" in item["extra"]]
            },
            "filename": [item["filename"] for item in batch]
        }

        if "rgbd_intrinsic" in batch[0]:
            collated_batch["rgbd_intrinsic"] = [item["rgbd_intrinsic"] for item in batch]
        if "rgbd_frames" in batch[0]:
            collated_batch["rgbd_frames"] = [item["rgbd_frames"] for item in batch]

        return collated_batch

class IGGFruitDatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.train_dataset = self.load_or_create_dataset(split='train')
        self.val_dataset = self.load_or_create_dataset(split='val')
        self.test_dataset = self.load_or_create_dataset(split='test')
        print("三个数据集已上传")

    def load_or_create_dataset(self, split):
        dataset_save_path = os.path.join(self.cfg.PATH, f'{split}_dataset.pkl')

        if os.path.exists(dataset_save_path):
            # 如果存在预处理后的数据集，则从文件中加载
            with open(dataset_save_path, 'rb') as f:
                # 确保 IGGFruit 类已经被定义
                dataset = pickle.load(f)
            print(f"从 {dataset_save_path} 加载 {split} 数据集")
        else:
            # 创建数据集并保存到文件
            dataset = IGGFruit(
                data_source=self.cfg.PATH,
                split=split,
                return_pcd=True,
                return_rgbd=True
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
            for key, value in sample.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  {sub_key}: {type(sub_value)}")
                else:
                    print(f"{key}: {type(value)}")

        return dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=self.train_dataset.collate_fn,
            shuffle=not self.cfg.MODEL.OVERFIT,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            collate_fn=self.val_dataset.collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            collate_fn=self.test_dataset.collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )

if __name__ == "__main__":
    # 定义数据源路径
    data_source = '/home/sweetpepper_dataset/shape_completion_challenge'

    # 示例配置类（用你的实际配置替换）
    class Config:
        PATH = data_source
        TRAIN = type('TRAIN', (), {'BATCH_SIZE': 4, 'NUM_WORKERS': 2})
        MODEL = type('MODEL', (), {'OVERFIT': False})

    cfg = Config()

    # 创建数据模块
    data_module = IGGFruitDatasetModule(cfg)

    # 示例检查训练数据加载器
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        print(batch)
        break

    # 检查测试数据加载器
    test_loader = data_module.test_dataloader()
    for batch in test_loader:
        print(batch)
        break

    import open3d as o3d

    # 从数据集中检索一个样本（此处为第一个）
    sample = data_module.train_dataset[0]

    # 从样本中提取下采样的点云数据
    points = sample["points"]
    colors = sample["colors"]

    # 创建一个 Open3D 点云对象并设置点和颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化下采样点云
    o3d.visualization.draw_geometries([pcd])


