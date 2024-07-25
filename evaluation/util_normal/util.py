from torch.utils.data import DataLoader
from .dataset_loader.dataset_loader_scannet import ScannetDataset
from .dataset_loader.dataset_loader_nyud import NYUD_Dataset
# from dataset_loader.dataset_loader_kinectazure import KinectAzureDataset
# from .dataset_loader.dataset_loader_scannet import Rectified2DOF
# from .dataset_loader.dataset_loader_scannet import Full2DOF
import numpy as np
import torch
import pdb
import os.path as osp

# def compute_surface_normal_angle_error(gt_norm, output_pred, mode='evaluate'):
#     surface_normal_pred = output_pred
#     if mode == 'evaluate':
#         prediction_error = torch.cosine_similarity(surface_normal_pred, gt_norm)
#         prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
#         return torch.acos(prediction_error) * 180.0 / np.pi

# def accumulate_prediction_error(total_normal_errors, gt_norm_mask, angle_error_prediction):
#     mask = gt_norm_mask > 0
#     if total_normal_errors is None:
#         total_normal_errors = angle_error_prediction[mask].data.cpu().numpy()
#     else:
#         total_normal_errors = np.concatenate((total_normal_errors, angle_error_prediction[mask].data.cpu().numpy()))

#     return total_normal_errors

# def log(str, fp=None):
#     if fp is not None:
#         fp.write('%s\n' % (str))
#         fp.flush()
#     print(str)


# def check_nan_ckpt(cnn):
#     is_nan_flag = False
#     for name, param in cnn.named_parameters():
#         if torch.sum(torch.isnan(param.data)):
#             is_nan_flag = True
#             break
#     return is_nan_flag

# def log_normal_stats(epoch, iter, normal_error_in_angle, fp=None):
#     log('Epoch %d, Iter %d, Mean %f, Median %f, Rmse %f, 5deg %f, 7.5deg %f, 11.25deg %f, 22.5deg %f, 30deg %f' %
#     (epoch, iter, np.average(normal_error_in_angle), np.median(normal_error_in_angle),
#      np.sqrt(np.sum(normal_error_in_angle * normal_error_in_angle) / normal_error_in_angle.shape),
#      np.sum(normal_error_in_angle < 5) / normal_error_in_angle.shape[0],
#      np.sum(normal_error_in_angle < 7.5) / normal_error_in_angle.shape[0],
#      np.sum(normal_error_in_angle < 11.25) / normal_error_in_angle.shape[0],
#      np.sum(normal_error_in_angle < 22.5) / normal_error_in_angle.shape[0],
#      np.sum(normal_error_in_angle < 30) / normal_error_in_angle.shape[0]), fp)

#     print('%f %f %f %f %f %f %f %f' %
#         (np.average(normal_error_in_angle), np.median(normal_error_in_angle),
#         np.sqrt(np.sum(normal_error_in_angle * normal_error_in_angle) / normal_error_in_angle.shape),
#         np.sum(normal_error_in_angle < 5) / normal_error_in_angle.shape[0],
#         np.sum(normal_error_in_angle < 7.5) / normal_error_in_angle.shape[0],
#         np.sum(normal_error_in_angle < 11.25) / normal_error_in_angle.shape[0],
#         np.sum(normal_error_in_angle < 22.5) / normal_error_in_angle.shape[0],
#         np.sum(normal_error_in_angle < 30) / normal_error_in_angle.shape[0]))

def create_dataset_loader(config, dataset_root, processing_res=0):
    # Testing on NYUD
    if config['TEST_DATASET'] == 'nyud':

        test_dataset = NYUD_Dataset(root=dataset_root, processing_res=processing_res)
        test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                                     shuffle=False, num_workers=4)

        return test_dataloader

    # # Testing on KinectAzure
    # if 'kinect_azure' in config['TEST_DATASET']:

    #     if config['TEST_DATASET'] == 'kinect_azure_full':
    #         test_dataset = KinectAzureDataset(usage='test_full')
    #     elif config['TEST_DATASET'] == 'kinect_azure_gravity_align':
    #         test_dataset = KinectAzureDataset(usage='test_gravity_align')
    #     elif config['TEST_DATASET'] == 'kinect_azure_tilted':
    #         test_dataset = KinectAzureDataset(usage='test_tilted')

    #     test_dataloader = DataLoader(test_dataset, batch_size=1,
    #                                  shuffle=False, num_workers=16)

    #     return test_dataloader

    # ScanNet standard split
    if 'standard' in config['TEST_DATASET']:

        test_dataset = ScannetDataset(root=dataset_root, usage='test', train_test_split=config['TEST_DATASET'], processing_res=processing_res)
        test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                                     shuffle=False, num_workers=4)

        return test_dataloader

    # # rectified_2dofa_scannet/framenet
    # if 'rectified_2dof' in config['TEST_DATASET']:

    #     test_dataset = Rectified2DOF(usage='test', train_test_split=config['TEST_DATASET'])
    #     test_dataloader = DataLoader(test_dataset, batch_size=1,
    #                                    shuffle=True, num_workers=16)
        
    #     return test_dataloader

    # # full_2dof_scannet/framenet
    # if 'full_2dof' in config['TEST_DATASET']:

    #     test_dataset = Full2DOF(usage='test', train_test_split=config['TEST_DATASET'])
    #     test_dataloader = DataLoader(test_dataset, batch_size=1,
    #                                   shuffle=True, num_workers=16, pin_memory=True)
        
        return test_dataloader

    return test_dataloader

if __name__ == '__main__':
    config = {'TEST_DATASET': 'nyud'}
    dataloader = create_dataset_loader(config)

    for sample_batched in dataloader:
        sample_batched = {data_key:sample_batched[data_key] for data_key in sample_batched}

        pdb.set_trace()
    a=1