import open3d as o3d
import numpy as np

from .utils import nn_correspondance

class ReconEvaluator:
    def __init__(self, threshold=.05, down_sample=.02):
        # self.threshold = threshold
        self.down_sample = down_sample

    def create_pcd_from_rgbd(self, images, scaled_depths, cam_intrs, poses, down_sample=False):
        
        # scaled_depths = scaled_depths.astype(np.float16)
        # cam_intrs = cam_intrs.astype(np.float16)

        # assert images.shape[0] == scaled_depths.shape[0] == poses.shape[0]
        # n_imgs, _, h, w = images.shape
        _, h, w = images.shape
        # if (len(cam_intrs.shape) == 3) and (cam_intrs.shape[-2:] == (3, 3)) and (cam_intrs.shape[0] == n_imgs):
        #     pass
        # elif (len(cam_intrs.shape) == 2) and (cam_intrs.shape == (3, 3)):
        #     cam_intrs = np.repeat(cam_intrs[None, ...], n_imgs, axis=0)
        # else:
        #     raise ValueError
                
        pcds = o3d.geometry.PointCloud()
        # for i in range(n_imgs):
        image = images
        depth = scaled_depths
        image = o3d.geometry.Image((image).astype(np.uint8))
        depth = o3d.geometry.Image((np.ascontiguousarray(depth.cpu()).astype(np.float32)))
        # import pdb
        # pdb.set_trace()
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(image, depth, depth_scale=1., depth_trunc=1e8, convert_rgb_to_intensity=False)
        intrinsic = cam_intrs
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
        pose = np.linalg.inv(poses)
        # pose = pose.astype(np.float16)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic, pose)
        if down_sample:
            pcd = pcd.voxel_down_sample(self.down_sample)
        pcds += pcd
        return pcds
    
    def pcd_match_ICP(self, pcd_pred, pcd_target):
        threshold = 1e8
        voxel_size = self.down_sample
        trans_init = np.eye(4)
        pcd_pred_temp = pcd_pred.voxel_down_sample(voxel_size)
        pcd_target_temp = pcd_target.voxel_down_sample(voxel_size)
        reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd_pred_temp, pcd_target_temp, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
        return reg_p2p

    def eval_pcd(self, file_pred, file_trgt):
        """ Compute Mesh metrics between prediction and target.
        Opens the Meshs and runs the metrics
        Args:
            file_pred: file path of prediction
            file_trgt: file path of target
            threshold: distance threshold used to compute precision/recal
            down_sample: use voxel_downsample to uniformly sample mesh points
        Returns:
            Dict of mesh metrics
        """

        if type(file_pred) == type(file_trgt) == str:
            pcd_pred = o3d.io.read_point_cloud(file_pred)
            pcd_trgt = o3d.io.read_point_cloud(file_trgt)
        elif type(file_pred) == type(file_pred) == o3d.geometry.PointCloud:
            pcd_pred = file_pred
            pcd_trgt = file_trgt
        else:
            raise ValueError
        if self.down_sample:
            pcd_pred = pcd_pred.voxel_down_sample(self.down_sample)
            pcd_trgt = pcd_trgt.voxel_down_sample(self.down_sample)
        verts_pred = np.asarray(pcd_pred.points)
        verts_trgt = np.asarray(pcd_trgt.points)

        _, dist1 = nn_correspondance(verts_pred, verts_trgt)
        _, dist2 = nn_correspondance(verts_trgt, verts_pred)
        dist1 = np.array(dist1) # completness
        dist2 = np.array(dist2) # accuracy

        # fscore
        precision = np.mean((dist2 < 0.05).astype('float'))
        recal = np.mean((dist1 < 0.05).astype('float'))
        fscore_5cm = 2 * precision * recal / (precision + recal + 1e-8)

        # precision = np.mean((dist2 < 0.1).astype('float'))
        # recal = np.mean((dist1 < 0.1).astype('float'))
        # fscore_10cm = 2 * precision * recal / (precision + recal)

        # precision = np.mean((dist2 < 0.2).astype('float'))
        # recal = np.mean((dist1 < 0.2).astype('float'))
        # fscore_20cm = 2 * precision * recal / (precision + recal)

        precision = np.mean((dist2 < 0.25).astype('float'))
        recal = np.mean((dist1 < 0.25).astype('float'))
        fscore_25cm = 2 * precision * recal / (precision + recal)

        # precision = np.mean((dist2 < 0.3).astype('float'))
        # recal = np.mean((dist1 < 0.3).astype('float'))
        # fscore_30cm = 2 * precision * recal / (precision + recal)

        # precision = np.mean((dist2 < 0.4).astype('float'))
        # recal = np.mean((dist1 < 0.4).astype('float'))
        # fscore_40cm = 2 * precision * recal / (precision + recal)

        precision = np.mean((dist2 < 0.5).astype('float'))
        recal = np.mean((dist1 < 0.5).astype('float'))
        fscore_50cm = 2 * precision * recal / (precision + recal)

        precision = np.mean((dist2 < 1).astype('float'))
        recal = np.mean((dist1 < 1).astype('float'))
        fscore_100cm = 2 * precision * recal / (precision + recal)

        metrics = {'dist1': np.mean(dist2),
                'dist2': np.mean(dist1),
                'c_L1': 0.5 * (np.mean(dist1) + np.mean(dist2)),
                'prec': precision,
                'recal': recal,
                'fscore_5cm': fscore_5cm,
                'fscore_25cm': fscore_25cm,
                'fscore_50cm': fscore_50cm,
                'fscore_100cm': fscore_100cm,
                # 'fscore_10cm': fscore_10cm,
                # 'fscore_20cm': fscore_20cm,
                # 'fscore_30cm': fscore_30cm,
                # 'fscore_40cm': fscore_40cm,
                # 'fscore_50cm': fscore_50cm,
                }
        return metrics

