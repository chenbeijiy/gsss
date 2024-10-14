    def atomize1(self):
        dist = self.get_scaling
        N = 2
        atom_mask = torch.min(self.get_scaling, dim=1).values <= self.atom_scale
        atom_mask1 = (torch.max(self.get_scaling, dim=1).values/torch.min(self.get_scaling, dim=1).values) > 2
        selected_pts_mask = torch.logical_and(atom_mask,atom_mask1)
        dist[selected_pts_mask] = self.atom_scale
        self.atom_scale*=0.99
        dist[atom_mask]*=0.99
        
        scaling_new = torch.log(dist)
        optimizable_tensors = self.replace_tensor_to_optimizer(scaling_new, "scaling")
        self._scaling = optimizable_tensors["scaling"]

        new_xyz = self.get_xyz[selected_pts_mask].repeat(N, 1)    # 将旋转后的样本点添加到原始点的位置。
        # 由于原来的3D gaussian的尺度过大, 现在将3D gaussian的尺度缩小为原来的1/1.6
        new_scaling = torch.log(dist)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)      # 将旋转矩阵重复 N 次。   # (2 * P, 4)

        # 将原始点的特征重复 N 次。
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)  # (2 * P, 1, 3)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)     # (2 * P, 15, 3)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)   # (2 * P, 1)

        # 调用另一个方法 densification_postfix，该方法对新生成的点执行后处理操作（此处跟densify_and_clone一样）。
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)    # 根据修剪过滤器，修剪模型中的一些参数。
