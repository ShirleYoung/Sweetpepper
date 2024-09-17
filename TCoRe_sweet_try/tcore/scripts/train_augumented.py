import os
import subprocess
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
import sys
import importlib.util
import inspect
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


TCoRe = import_module_from_file(
    "tcore_sweet_try.models.model",
    "tcore/models/model.py"
).TCoRe

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
import importlib
import sys
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

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
            gt_pcd = self.get_gt(fid)
            input_data = self.get_rgbd(fid)

            points = np.asarray(input_data['pcd'].points)
            colors = np.asarray(input_data['pcd'].colors)
            gt_points = np.asarray(gt_pcd.points)
            gt_normals = np.asarray(gt_pcd.normals)

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
                    "gt_points": gt_points,
                    "gt_normals": gt_normals
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
        print("Shape of gt_points:", item["extra"]["gt_points"].shape)
        return item

    @staticmethod
    def collate_fn(batch):
        collated_batch = {
            "points": [item["points"] for item in batch],
            "colors": [item["colors"] for item in batch],
            "extra": {
                "gt_points": [item["extra"]["gt_points"] for item in batch],
                "gt_normals": [item["extra"]["gt_normals"] for item in batch]
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
        print("三个数据集已上传")

    def load_or_create_dataset(self, split):
        print(os.path.abspath(__file__))
        print(os.getcwd())
        if split == 'train':
            augmented_dataset_save_path = '../sweetpepper_dataset/shape_completion_challenge/train_augmented_dataset.pkl'
            absolute_path = os.path.abspath(augmented_dataset_save_path)
            print(absolute_path)

            # 检查是否存在增强后的数据集
            if os.path.exists(augmented_dataset_save_path):
                with open(augmented_dataset_save_path, 'rb') as f:
                    dataset = pickle.load(f)
                print(f"从 {augmented_dataset_save_path} 加载增强后的训练数据集")
                return dataset
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

        augmented_dataset=dataset
        # 新增数据增强部分
        # 新增数据增强部分
        if split == 'train':
            print(f"正在对 {split} 数据集进行数据增强...")
            
            # 用于暂存增强样本的列表
            new_samples = []

            for fid, sample in dataset.preprocessed_data.items():
                original_points = sample["points"]
                original_colors = sample["colors"]

                # 将原始样本下采样到 45000 个点
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(original_points)
                pcd.colors = o3d.utility.Vector3dVector(original_colors)
                downsampled_pcd = IGGFruit.fps_downsample(pcd, 35000)
                downsampled_points = np.asarray(downsampled_pcd.points)
                downsampled_colors = np.asarray(downsampled_pcd.colors)

                # 创建第一个增强样本：在下采样后的点云上添加 5000 个随机点
                random_points = np.random.uniform(low=downsampled_points.min(axis=0), 
                                                high=downsampled_points.max(axis=0), 
                                                size=(5000, 3))
                random_colors = np.random.uniform(low=0, high=1, size=(5000, 3))
                augmented_points_add = np.vstack([downsampled_points, random_points])
                augmented_colors_add = np.vstack([downsampled_colors, random_colors])

                print(f"增强后的points shape为：{augmented_points_add.shape}")

                # 将增强样本暂存到 new_samples 列表
                new_samples.append((f"{fid}_add_noise", {
                    "points": augmented_points_add,
                    "colors": augmented_colors_add,
                    "extra": sample["extra"],
                    "filename": f"{sample['filename']}_add_noise"
                }))

            
            # 遍历完成后将新样本添加到字典中
            for new_fid, new_sample in new_samples:
                augmented_dataset.preprocessed_data[new_fid] = new_sample
                augmented_dataset.fruit_list[new_fid] = {"path": new_sample["filename"]}

            dataset = augmented_dataset
            print(f"{split} 数据集数据增强完成，现在的长度为 {len(dataset)}")
                    # 将增强后的数据集保存到指定文件
            augmented_dataset_save_path = '../sweetpepper_dataset/shape_completion_challenge/train_augmented_dataset.pkl'
            with open(augmented_dataset_save_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"将增强后的训练数据集保存到 {augmented_dataset_save_path}")

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

@click.command()
@click.option("--w", type=str, required=False)
@click.option("--ckpt_path", type=str, default=None, required=False)  # 添加的参数
@click.option("--ckpt", type=str, default=None, required=False)
@click.option("--bb_cr", type=float, default=None, required=False)
@click.option("--dec_cr", type=float, default=None, required=False)
@click.option("--dec_blocks", type=int, default=None, required=False)
@click.option("--iterative", is_flag=True)
@click.option("--model_cfg_path", type=str, default="../config/model.yaml", required=False)
def main(w, ckpt, bb_cr, dec_cr, dec_blocks, iterative, model_cfg_path, ckpt_path):#这里也修改了
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), model_cfg_path)))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    decoder_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    cfg.git_commit_version = str(
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip()
    )

    if cfg.MODEL.DATASET == "FRUITS":
        print("Using FRUITS dataset")
        print("IGGFruitDatasetModule begins")
        data = IGGFruitDatasetModule(cfg)
        print("IGGFruitDatasetModule ends")
    else:
        raise NotImplementedError

    if bb_cr:
        cfg.BACKBONE.CR = bb_cr
    if dec_cr:
        cfg.DECODER.CR = dec_cr
    if dec_blocks:
        cfg.DECODER.DEC_BLOCKS = dec_blocks

    if iterative:
        cfg.DECODER.ITERATIVE_TEMPLATE = True
    model = TCoRe(cfg)
    if w:
        w = torch.load(w, map_location="cpu")
        model.load_state_dict(w["state_dict"])

    if ckpt_path:  # 使用新参数加载预训练模型权重，这是新加的
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    tb_logger = pl_loggers.TensorBoardLogger(
        cfg.LOGDIR + cfg.EXPERIMENT.ID, default_hp_metric=False
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")

    cd_ckpt = ModelCheckpoint(
        monitor="val_chamfer_distance",
        filename=cfg.EXPERIMENT.ID +
        "_epoch{epoch:02d}_cd{val_chamfer_distance:.2f}",
        auto_insert_metric_name=False,
        mode="min",
        save_last=True,
    )

    precision_ckpt = ModelCheckpoint(
        monitor="val_precision_auc",
        filename=cfg.EXPERIMENT.ID +
        "_epoch{epoch:02d}_pr{val_precision_auc:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )

    recall_ckpt = ModelCheckpoint(
        monitor="val_recall_auc",
        filename=cfg.EXPERIMENT.ID +
        "_epoch{epoch:02d}_re{val_recall_auc:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )

    fscore_ckpt = ModelCheckpoint(
        monitor="val_fscore_auc",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_f{val_fscore_auc:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )

    trainer = Trainer(
        num_sanity_val_steps=0,
        gpus=cfg.TRAIN.N_GPUS,
        accelerator="cuda",
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=[lr_monitor, fscore_ckpt,
                   precision_ckpt, recall_ckpt, cd_ckpt],
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        resume_from_checkpoint=ckpt,
        check_val_every_n_epoch=1,
    )

    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = total_memory - reserved_memory

    print("CUDA显存总容量: {:.2f} GiB".format(total_memory / (1024 ** 3)))
    print("CUDA已分配内存: {:.2f} GiB".format(allocated_memory / (1024 ** 3)))
    print("CUDA已保留内存: {:.2f} GiB".format(reserved_memory / (1024 ** 3)))
    print("CUDA剩余可用内存: {:.2f} GiB".format(free_memory / (1024 ** 3)))

    print("Training begins")
    trainer.fit(model, data)
    trainer.test(model, dataloaders=data.val_dataloader())

def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))

if __name__ == "__main__":
    main()
