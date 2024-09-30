import numpy as np
from plyfile import PlyData, PlyElement
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_ply(path):
        plydata = PlyData.read(path)


        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])


        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        scales = np.exp(scales)

        data = np.sum(scales,axis=1,keepdims=True)

        data_min = np.min(data)
        data_max = np.max(data)
        normalized_data = (data - data_min) / (data_max - data_min)

        ount_zeros = np.sum(normalized_data == 0)

        return data

    
data = load_ply("./test/point_cloud.ply")

# 将数据展平为一维数组
data_flat = data.flatten()

# 统计每个值的出现次数
unique_values, counts = np.unique(data_flat, return_counts=True)

# 创建一个包含值和对应数量的字典
value_count_dict = {float(value): count for value, count in zip(unique_values, counts)}

# 按数量排序并保留前10个
sorted_counts = sorted(value_count_dict.items(), key=lambda x: x[1], reverse=True)[:10]

# 输出前10个数量最多的值
print("前10个数量最多的值及其数量:")
for value, count in sorted_counts:
    print(f"{value}: {count}")
