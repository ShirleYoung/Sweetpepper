import open3d as o3d
import numpy as np
import os

# 设置文件夹路径
folder_a = r"./pointft_dataset/test"
folder_b = r"./pointft_result"
output_folder = r"./submission"

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历folder_a中的所有.ply文件
for filename in os.listdir(folder_a):
    if filename.endswith(".ply"):
        # 读取点云A和点云B
        path_a = os.path.join(folder_a, filename)
        path_b = os.path.join(folder_b, filename)
        
        if os.path.exists(path_b):
            pcd_a = o3d.io.read_point_cloud(path_a)
            pcd_b = o3d.io.read_point_cloud(path_b)
            
            # 使用KDTree加速最近邻搜索
            pcd_a_tree = o3d.geometry.KDTreeFlann(pcd_a)
            
            # 过滤点云B中的点
            filtered_points = []
            for point in np.asarray(pcd_b.points):
                # 查询点云A中与当前点最近的所有点
                [k, idx, _] = pcd_a_tree.search_radius_vector_3d(point, 0.002)
                
                # 如果至少有5个点距离小于0.002，则保留该点
                if k >= 5:
                    filtered_points.append(point)
            
            # 创建新的点云B，保留过滤后的点
            pcd_b_filtered = o3d.geometry.PointCloud()
            pcd_b_filtered.points = o3d.utility.Vector3dVector(filtered_points)
            
            # 融合点云A和过滤后的点云B
            pcd_combined = pcd_a + pcd_b_filtered
            
            # 生成输出路径
            output_path = os.path.join(output_folder, filename)
            
            # 保存结果
            o3d.io.write_point_cloud(output_path, pcd_combined)
            
            # 输出点的总数
            print(f"Processed {filename}")
