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
        points = item["points"]  # Ensure points is (num_points, 3)
        gt_points = item["gt_points"]  # Ensure gt_points is (num_points, 3) or an empty array
        label = 0  # Set a fixed label or modify it as per your requirements
        if self.split != 'test':
            return label, points, gt_points
        else:
            return label, points, points

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
import torch
from utils.train_utils import *
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

def train():
    logging.info(str(args))
    metrics = ['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    # 初始化数据集
    if args.dataset == 'iggfruit':
        dataset = load_or_create_dataset(split='train')
        dataset_test = load_or_create_dataset(split='val')
    else:
        raise ValueError('dataset is not exist')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))

    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))
    
    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)
    
    lr = args.lr
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]
    
    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.module.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)
    
    for epoch in range(args.start_epoch, args.nepoch):
    
        train_loss_meter.reset()
        net.module.train()

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
    
            _, inputs, gt = data
            # mean_feature = None
    
            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()
            print(f"Batch x: inputs shape = {inputs.shape}")
    
            out2, loss2, net_loss = net(inputs, gt)
    
            train_loss_meter.update(net_loss.mean().item())
    
            net_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
    
            optimizer.step()
    
            if i % args.step_interval_to_print == 0:
                logging.info(exp_name + ' train [%d: %d/%d]  loss_type: %s, fine_loss: %f total_loss: %f lr: %f' %
                             (epoch, i, len(dataset) / args.batch_size, args.loss, loss2.mean().item(), net_loss.mean().item(), lr))
    
        if epoch % args.epoch_interval_to_save == 0:
            save_path = '%s/network.pth' % log_dir
            save_model('%s/network.pth' % log_dir, net)
            logging.info("Saving net...")
            print(f"Model saved at: {save_path}")  # 新增代码：输出保存路径
    
        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses)


def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses):
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            label, inputs, gt = data
    
            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()
            result_dict = net(inputs, gt, is_training=False)
            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item())
    
        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)
    
        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)
    
        logging.info(curr_log)
        logging.info(best_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    print(os.getcwd())
    config_path = 'cfgs/PointFT.yaml'
    absolute_path = os.path.abspath(config_path)
    print(absolute_path)

    args = munch.munchify(yaml.safe_load(open(config_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    
    else:
        exp_name = args.model_name + '_' + args.loss + '_' + args.flag + '_' + args.dataset
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    print('save_path:', args.work_dir)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)],
                        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
                        force=True)
    train()
