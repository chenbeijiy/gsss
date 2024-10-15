# import torch
# import os
# import numpy as np
# from plyfile import PlyData, PlyElement

# def atomize1(self):#, levels_lod):
#         # dist = self.get_scaling
#         # atom_scale = self.atom_scale_all[levels_lod]
            
#         N = 2
#         atom_mask = torch.min(self.get_scaling, dim=1).values < self.atom_scale
#         atom_mask1 = (torch.max(self.get_scaling, dim=1).values/torch.min(self.get_scaling, dim=1).values) > 5
#         selected_pts_mask = torch.logical_and(atom_mask,atom_mask1)
#         print("number:",torch.sum(selected_pts_mask).item())
#         # dist[selected_pts_mask] = self.atom_scale
        
#         # scaling_new = torch.log(dist)
#         # optimizable_tensors = self.replace_tensor_to_optimizer(scaling_new, "scaling")
#         # self._scaling = optimizable_tensors["scaling"]

#         scaling = self.get_scaling[selected_pts_mask]
#         # atom_selected_scales = atom_scale[selected_pts_mask]
#         # atom_selected_scales = atom_selected_scales.unsqueeze(1).repeat(1, 2)
#         rots = build_rotation(self._rotation[selected_pts_mask])
#         dirs, max_scaling, axis = self.get_dir_max_scaling(scaling, rots)  # 获取最大轴方向
#         radii = (2 * max_scaling / 3.)[..., None] # 3 std
#         new_xyz1 = self.get_xyz[selected_pts_mask] + dirs * radii  # 新高斯位置均匀划分旧高斯最大尺度
#         new_xyz2 = self.get_xyz[selected_pts_mask] - dirs * radii
#         new_xyz = torch.cat((new_xyz1, new_xyz2), dim=0)
#         # new_scaling = torch.log(dist).repeat(N,1)
#         new_scaling = scaling.detach().clone()
#         # new_scaling[:] = atom_selected_scales
#         new_scaling[:] = self.atom_scale
#         new_scaling = self.scaling_inverse_activation(new_scaling)
#         new_scaling = torch.cat((new_scaling, new_scaling), dim=0)

#         new_rotation = self._rotation[selected_pts_mask].repeat(N,1)      # 将旋转矩阵重复 N 次。   # (2 * P, 4)

#         # 将原始点的特征重复 N 次。
#         new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)  # (2 * P, 1, 3)
#         new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)     # (2 * P, 15, 3)
#         new_opacity = self._opacity[selected_pts_mask].repeat(N,1)   # (2 * P, 1)

#         # 调用另一个方法 densification_postfix，该方法对新生成的点执行后处理操作（此处跟densify_and_clone一样）。
#         self.atom_scale*=0.98
#         # dist[atom_mask]*=0.99
#         self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

#         prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
#         self.prune_points(prune_filter)    # 根据修剪过滤器，修剪模型中的一些参数。

#         dist = self.get_scaling
#         scaling_new = torch.log(dist)
#         optimizable_tensors = self.replace_tensor_to_optimizer(scaling_new, "scaling")
#         self._scaling = optimizable_tensors["scaling"]

# def load_ply(self, path):
#     plydata = PlyData.read(path)

#     xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
#                     np.asarray(plydata.elements[0]["y"]),
#                     np.asarray(plydata.elements[0]["z"])),  axis=1)
#     opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

#     features_dc = np.zeros((xyz.shape[0], 3, 1))
#     features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
#     features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
#     features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

#     extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
#     extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
#     assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
#     features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
#     for idx, attr_name in enumerate(extra_f_names):
#         features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
#     # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
#     features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

#     scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
#     scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
#     scales = np.zeros((xyz.shape[0], len(scale_names)))
#     for idx, attr_name in enumerate(scale_names):
#         scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

#     rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
#     rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
#     rots = np.zeros((xyz.shape[0], len(rot_names)))
#     for idx, attr_name in enumerate(rot_names):
#         rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

#     self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
#     self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
#     self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
#     self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
#     self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
#     self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

#     self.active_sh_degree = self.max_sh_degree

# def get_dir_max_scaling(self, scaling, rots):
#         '''
#             rots: N, 3, 3
#         '''
#         axis = torch.argmax(scaling, dim=-1)
#         max_scaling = scaling[torch.arange(scaling.shape[0]), axis]
#         dirs = rots.gather(2, axis[:, None, None].expand(-1, 3, -1)).squeeze(-1)
        
#         return dirs, max_scaling, axis

# def build_rotation(r):

#     norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

#     q = r / norm[:, None]

#     R = torch.zeros((q.size(0), 3, 3), device='cuda')

#     r = q[:, 0]
#     x = q[:, 1]
#     y = q[:, 2]
#     z = q[:, 3]

#     R[:, 0, 0] = 1 - 2 * (y*y + z*z)
#     R[:, 0, 1] = 2 * (x*y - r*z)
#     R[:, 0, 2] = 2 * (x*z + r*y)
#     R[:, 1, 0] = 2 * (x*y + r*z)
#     R[:, 1, 1] = 1 - 2 * (x*x + z*z)
#     R[:, 1, 2] = 2 * (y*z - r*x)
#     R[:, 2, 0] = 2 * (x*z - r*y)
#     R[:, 2, 1] = 2 * (y*z + r*x)
#     R[:, 2, 2] = 1 - 2 * (x*x + y*y)
#     return R


# gaussian.load_ply(os.path.join(self.model_path,
#                                                 "point_cloud",
#                                                     "iteration_" + str(self.loaded_iter),
#                                                     "point_cloud.ply"))