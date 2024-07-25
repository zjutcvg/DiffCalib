# TODO

mldb_info = {}

#### Taskonomy Dataset

mldb_info['Taskonomy']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'Taskonomy',
    'semantic_root': 'Taskonomy',
    'meta_data_root': 'Taskonomy',
    'norm_root': 'Taskonomy',
    'mask_feature': 'Taskonomy/mask_feature',
    'train_annotations_path': 'Taskonomy/Taskonomy/train_annotation_April_all_new.json',
    'finetune_annotations_path': 'Taskonomy/Taskonomy/train_annotation_diversity_direct.json',
    # 'train_annotations_path': 'Taskonomy/train_annotation_April.json',
    # 'test_annotations_path': 'Taskonomy/train_annotation_April.json',
    # 'val_annotations_path': 'Taskonomy/Taskonomy/val_annotations_100.json',
    'val_annotations_path': 'Taskonomy/train_annotation_April.json',
    'metric_scale': 512.0
}
mldb_info['Metric3D']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'Metric3D',
    'semantic_root': 'Metric3D',
    'meta_data_root': 'Metric3D',
    'norm_root': 'Metric3D',
    'mask_feature': 'Metric3D/mask_feature',
    'train_annotations_path': 'Metric3D/train_annotations.json',
    # 'train_annotations_path': 'Taskonomy/Taskonomy/train_annotations_15w.json',
    # 'train_annotations_path': 'Taskonomy/train_annotation_April.json',
    # 'test_annotations_path': 'Taskonomy/train_annotation_April.json',
    # 'val_annotations_path': 'Taskonomy/Taskonomy/val_annotations_100.json',
    # 'val_annotations_path': 'Taskonomy/train_annotation_April.json',
    'metric_scale': 1.0
}

mldb_info['Lyft']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'lyft',
    'semantic_root': 'lyft',
    'meta_data_root': 'lyft',
    'norm_root': 'lyft',
    'train_annotations_path': 'lyft/lyft/train_annotation_April_all.json',
    'test_annotations_path': 'lyft/lyft/train_annotation_April_all.json',
    'val_annotations_path': 'lyft/lyft/train_annotation_April_all.json',
    'ffrecord_root': '/mnt/nas/share/home/jxr/MetricDepth/ffrecord/ffrecord/lyft',
    'metric_scale': 300.0
}

mldb_info['DDAD']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'DDAD',
    'semantic_root': 'DDAD',
    'meta_data_root': 'DDAD',
    'norm_root': 'DDAD',
    'train_annotations_path': 'DDAD/DDAD/train/train_annotation_April_all.json',
    'test_annotations_path': 'DDAD/test_annotation.json',
    'val_annotations_path': 'DDAD/DDAD/train/train_annotation_April_all.json',
    'ffrecord_root': '/mnt/nas/share/home/jxr/MetricDepth/ffrecord/ffrecord/DDAD',
    'metric_scale': 300.0
}

mldb_info['DSEC']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'DSEC',
    'semantic_root': 'DSEC',
    'meta_data_root': 'DSEC',
    'norm_root': 'DSEC',
    'train_annotations_path': 'DSEC/DSEC/train_annotation_all_August.json',
    'test_annotations_path': 'DSEC/DSEC/train_annotation_all_August.json',
    'val_annotations_path': 'DSEC/DSEC/train_annotation_all_August.json',
    'ffrecord_root': '/mnt/nas/share/home/jxr/MetricDepth/ffrecord/new_ffrecord/DSEC',
    'metric_scale': 300.0
}

mldb_info['A2D2']={
    'mldb_root': '/mnt/nas/share/home/jxr/MetricDepth/',
    'data_root': 'A2D2',
    'semantic_root': 'A2D2',
    'meta_data_root': 'A2D2',
    'norm_root': 'A2D2',
    'train_annotations_path': 'A2D2/A2D2/train_annotation_April_all.json',
    'test_annotations_path': 'A2D2/A2D2/train_annotation_April_all.json',
    'val_annotations_path': 'A2D2/A2D2/train_annotation_April_all.json',
    'ffrecord_root': '/mnt/nas/share/home/jxr/MetricDepth/ffrecord/ffrecord/A2D2',
    'metric_scale': 512.0
}

mldb_info['UASOL']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'UASOL',
    'semantic_root': 'UASOL',
    'meta_data_root': 'UASOL',
    'norm_root': 'UASOL',
    'train_annotations_path': 'UASOL/UASOL/train_annotation_April_all.json',
    'test_annotations_path': 'UASOL/UASOL/train_annotation_April_all.json',
    'val_annotations_path': 'UASOL/UASOL/train_annotation_April_all.json',
    'ffrecord_root': '/mnt/nas/share/home/jxr/MetricDepth/ffrecord/ffrecord/UASOL',
    'metric_scale': 300.0
}

mldb_info['DIML']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'DIML',
    'semantic_root': 'DIML',
    'meta_data_root': 'DIML',
    'norm_root': 'DIML',
    'train_annotations_path': 'DIML/DIML/train_annotation_April_all.json',
    'test_annotations_path': 'DIML/DIML/train_annotation_April_all.json',
    'val_annotations_path': 'DIML/DIML/train_annotation_April_all.json',
    'ffrecord_root': '/mnt/nas/share/home/jxr/MetricDepth/ffrecord/ffrecord/DIML',
    'metric_scale': 300.0
}

mldb_info['Cityscapes']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'cityscapes',
    'semantic_root': 'cityscapes',
    'meta_data_root': 'cityscapes',
    'norm_root': 'cityscapes',
    'train_annotations_path': 'cityscapes/cityscapes/train_annotation_April_all_new.json',
    'test_annotations_path': 'cityscapes/cityscapes/train_annotation_April_all_new.json',
    'val_annotations_path': 'cityscapes/cityscapes/train_annotation_April_all_new.json',
    'ffrecord_root': '/mnt/nas/share/home/jxr/MetricDepth/ffrecord/new_ffrecord/cityscapes',
    'metric_scale': 300.0
}

mldb_info['Argovers2']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'argoverse',
    'semantic_root': 'argoverse',
    'meta_data_root': 'argoverse',
    'norm_root': 'argoverse',
    'train_annotations_path': 'argoverse/argoverse/train_annotation_April_all.json',
    'test_annotations_path': 'argoverse/argoverse/train_annotation_April_all.json',
    'val_annotations_path': 'argoverse/argoverse/train_annotation_April_all.json',
    'ffrecord_root': '/mnt/nas/share/home/jxr/MetricDepth/ffrecord/ffrecord/argoverse',
    'metric_scale': 300.0
}


mldb_info['Mapillary_PSD']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'mapillary',
    'semantic_root': 'mapillary',
    'meta_data_root': 'mapillary',
    'norm_root': 'mapillary',
    'train_annotations_path': 'mapillary/mapillary/train_annotation_April_all.json',
    'test_annotations_path': 'mapillary/mapillary/train_annotation_April_all.json',
    'val_annotations_path': 'mapillary/mapillary/train_annotation_April_all.json',
    'ffrecord_root': '/mnt/nas/share/home/jxr/MetricDepth/ffrecord/ffrecord/mapillary',
    'metric_scale': 256.0
}


mldb_info['Pandaset']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'pandaset',
    'semantic_root': 'pandaset',
    'meta_data_root': 'pandaset',
    'norm_root': 'pandaset',
    'train_annotations_path': 'pandaset/pandaset/train_annotation_April_all.json',
    'test_annotations_path': 'pandaset/pandaset/train_annotation_April_all.json',
    'val_annotations_path': 'pandaset/pandaset/train_annotation_April_all.json',
    'ffrecord_root': '/mnt/nas/share/home/jxr/MetricDepth/ffrecord/ffrecord/pandaset',
    'metric_scale': 200.0
}

mldb_info['DrivingStereo']={
    'mldb_root': '/mnt/nas/share/home/jxr/MetricDepth/',
    'data_root': 'drivingstereo',
    'semantic_root': 'drivingstereo',
    'meta_data_root': 'drivingstereo',
    'norm_root': 'drivingstereo',
    'train_annotations_path': 'drivingstereo/drivingstereo/train_annotation_April_all.json',
    'test_annotations_path': 'drivingstereo/drivingstereo/train_annotation_April_all.json',
    'val_annotations_path': 'drivingstereo/drivingstereo/train_annotation_April_all.json',
    'ffrecord_root': '/mnt/nas/share/home/jxr/MetricDepth/ffrecord/new_ffrecord/drivingstereo',
    'metric_scale': 512.0
}

mldb_info['Waymo']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'Waymo',
    'semantic_root': 'Waymo',
    'meta_data_root': 'Waymo',
    'norm_root': 'Waymo',
    'train_annotations_path': 'Waymo/Waymo/train_annotation_April_all_new.json',
    'test_annotations_path': 'Waymo/Waymo/train_annotation_April_all_new.json',
    'val_annotations_path': 'Waymo/Waymo/train_annotation_April_all_new.json',
    'metric_scale': 200.0
}

#### test datasets

mldb_info['Scannet']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    # 'mldb_root': '/mnt/nas/share/home/xugk/data/',
    'data_root': 'ScanNet_v2',
    'semantic_root': 'ScanNet_v2',
    'meta_data_root': 'ScanNet_v2',
    'norm_root': 'ScanNet_v2',

    'finetune_annotations_path': 'ScanNet_v2/train_annotation_diversity.json',
    'train_annotations_path': 'ScanNet_v2/train_annotations.json',
    'test_annotations_path': 'ScanNet_v2/train_annotations.json',
    'val_annotations_path': 'ScanNet_v2/train_annotations.json',
    'metric_scale': 1000.0
}

mldb_info['scannet_test']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',

    'data_root': 'scannet_test',
    'semantic_root': 'scannet_test',
    'meta_data_root': 'scannet_test',
    'norm_root': 'scannet_test',

    'finetune_annotations_path': 'scannet_test/train_annotation_pred_depth.json',
    'train_annotations_path': 'scannet_test/train_annotation.json',
    'test_annotations_path': 'scannet_test/test_annotation.json',
    'val_annotations_path': 'scannet_test/test_annotation.json',
    'metric_scale': 1000.0
}

mldb_info['KITTI']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'kitti',
    'semantic_root': 'kitti',
    'meta_data_root': 'kitti',
    'norm_root': 'kitti',
    'train_annotations_path': 'kitti/train_annotation.json',
    # 'train_annotations_path': 'kitti/test_annotation.json',
    'test_annotations_path': 'kitti/test_annotation.json',
    'val_annotations_path': 'kitti/test_annotation.json',
    'metric_scale': 256.0
}

mldb_info['NYU']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'nyu',
    'semantic_root': 'nyu',
    'meta_data_root': 'nyu',
    'norm_root': 'nyu',

    'train_annotations_path': 'nyu/train_annotation.json',
    # 'finetune_annotations_path': 'nyu/train_annotation_pred_depth.json',
    'finetune_annotations_path': 'nyu/train_annotation.json',
    'val_annotations_path': 'nyu/test_annotation.json',
    'test_annotations_path': 'nyu/test_annotation.json',
    'metric_scale': 1000.0
}

# mldb_info['NYU']={
#     'mldb_root': '/mnt/nas/share/home/xugk/',
#     'data_root': 'data_demo',
#     'semantic_root': 'data_demo',
#     'meta_data_root': 'data_demo',
#     'norm_root': 'data_demo',
#     'train_annotations_path': 'nyu/train_annotation.json',
#     'test_annotations_path': 'data_demo/test_annotation.json',
#     'val_annotations_path': 'nyu/test_annotation.json',
# }

mldb_info['SevenScenes']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': '7scenes',
    'semantic_root': '7scenes',
    'meta_data_root': '7scenes',
    'norm_root': '7scenes',
    'train_annotations_path': '7scenes/train_annotations.json',
    'test_annotations_path': '7scenes/test_annotation.json',
    'val_annotations_path': '7scenes/test_annotation.json',
    'metric_scale': 1.0

    # 'train_annotations_path': '7scenes/train_annotation_wo_chess.json',
    # 'test_annotations_path': '7scenes/test_annotation_chess.json',
    # 'val_annotations_path': '7scenes/test_annotation_chess.json',
}

mldb_info['DIODE']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'diode',
    'semantic_root': 'diode',
    'meta_data_root': 'diode',
    'norm_root': 'diode',
    'train_annotations_path': 'diode/train_annotations.json',
    # 'test_annotations_path': 'diode/test_annotation_outdoor.json',
    # 'test_annotations_path': 'diode/test_annotation_indoors.json',
    'test_annotations_path': 'diode/test_annotation.json',
    'val_annotations_path': 'diode/test_annotation.json',
    'metric_scale': 1.0
}

mldb_info['DIODE_indoor']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'diode',
    'semantic_root': 'diode',
    'meta_data_root': 'diode',
    'norm_root': 'diode',
    'train_annotations_path': 'diode/train_annotations_indoor.json',
    # 'finetune_annotations_path': 'diode/train_annotations_indoor.json',
    'finetune_annotations_path': 'diode/train_annotations_indoor.json',
    'val_annotations_path': 'diode/test_annotation_indoors.json',
    # 'val_annotations_path': 'diode/test_annotation_indoors.json',
    'test_annotations_path': 'diode/test_annotation_indoors.json',
    'metric_scale': 1.0
}

mldb_info['DIODE_outdoor']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'diode',
    'semantic_root': 'diode',
    'meta_data_root': 'diode',
    'norm_root': 'diode',
    'test_annotations_path': 'diode/test_annotation_outdoor.json',
    'metric_scale': 512.0
}

mldb_info['Nuscenes']={
    'mldb_root': '/mnt/nas/share/home/xugk/data/',
    'data_root': 'nuscenes',
    'semantic_root': 'nuscenes',
    'meta_data_root': 'nuscenes',
    'norm_root': 'nuscenes',
    'train_annotations_path': '',
    'test_annotations_path': 'nuscenes/test_annotation.json',
    'val_annotations_path': 'nuscenes/test_annotation.json',
    'metric_scale': 512.0
}

mldb_info['TUM']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'TUM',
    'semantic_root': 'TUM',
    'meta_data_root': 'TUM',
    'norm_root': 'TUM',
    'train_annotations_path': 'TUM/train_annotations.json',
    'finetune_annotations_path': 'TUM/train_annotations.json',
    'test_annotations_path': 'TUM/train_annotations.json',
    'val_annotations_path': 'TUM/train_annotations.json',
    'metric_scale': 5000.0
}

mldb_info['Hypersim']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'hypersim',
    'semantic_root': 'hypersim',
    'meta_data_root': 'hypersim',
    'norm_root': 'hypersim',
    'train_annotations_path': 'hypersim/train_annotations.json',
    # 'test_annotations_path': 'hypersim/train_annotations.json',
    'val_annotations_path': 'hypersim/train_annotations.json',
    'metric_scale': 1.0
}

mldb_info['GraspNet']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'graspnet',
    'semantic_root': 'graspnet',
    'meta_data_root': 'graspnet',
    'norm_root': 'graspnet',
    'train_annotations_path': 'graspnet/train_annotations.json',
    'finetune_annotations_path': 'graspnet/train_annotation_pred_depth.json',
    # 'test_annotations_path': 'graspnet/train_annotations.json',
    'val_annotations_path': 'graspnet/train_annotations.json',
    'metric_scale': 256.0
}

mldb_info['Tartanair']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'tartanair',
    'semantic_root': 'tartanair',
    'meta_data_root': 'tartanair',
    'norm_root': 'tartanair',
    'train_annotations_path': 'tartanair/train_annotations.json',
    'test_annotations_path': 'tartanair/train_annotations.json',
    'val_annotations_path': 'tartanair/train_annotations.json',
    'metric_scale': 1.0
}

mldb_info['AVD']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'AVD',
    'semantic_root': 'AVD',
    'meta_data_root': 'AVD',
    'norm_root': 'AVD',
    'train_annotations_path': 'AVD/train_annotations.json',
    # 'finetune_annotations_path': 'AVD/train_annotation_pred_depth.json',
    'test_annotations_path': 'AVD/train_annotations.json',
    # 'val_annotations_path': 'AVD/train_annotations.json',
    'val_annotations_path': 'AVD/train_annotation_pred_depth.json',
    'metric_scale': 1000.0
}

mldb_info['BlendedMVS']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'BlendedMVS',
    'semantic_root': 'BlendedMVS',
    'meta_data_root': 'BlendedMVS',
    'norm_root': 'BlendedMVS',
    'train_annotations_path': 'BlendedMVS/train_annotations.json',
    # 'test_annotations_path': 'BlendedMVS/train_annotations.json',
    'val_annotations_path': 'BlendedMVS/train_annotations.json',
    'metric_scale': 1.0
}

mldb_info['CO3D']={
    'mldb_root': '/mnt/nas/share/home/xugk/data/',
    'data_root': 'CO3D',
    'semantic_root': 'CO3D',
    'meta_data_root': 'CO3D',
    'norm_root': 'CO3D',
    'train_annotations_path': 'CO3D/train_annotations.json',
    # 'test_annotations_path': 'CO3D/train_annotations.json',
    'val_annotations_path': 'CO3D/train_annotations.json',
}

mldb_info['crestereo']={
    'mldb_root': '/mnt/nas/share/home/xugk/data/',
    'data_root': 'crestereo',
    'semantic_root': 'crestereo',
    'meta_data_root': 'crestereo',
    'norm_root': 'crestereo',
    'train_annotations_path': 'crestereo/train_annotations.json',
    # 'test_annotations_path': 'crestereo/train_annotations.json',
    'val_annotations_path': 'crestereo/train_annotations.json',
}

mldb_info['ETH3D']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'ETH3D',
    'semantic_root': 'ETH3D',
    'meta_data_root': 'ETH3D',
    'norm_root': 'ETH3D',
    'test_annotations_path': 'ETH3D/test_annotations.json',
    'val_annotations_path': 'ETH3D/test_annotations.json',
    'metric_scale': 1.0
}

mldb_info['ibims']={
    'mldb_root': '/test/xugk/data/data_metricdepth/',
    'data_root': 'ibims',
    'semantic_root': 'ibims',
    'meta_data_root': 'ibims',
    'norm_root': 'ibims',
    'train_annotations_path': 'ibims/test_annotations.json',
    'finetune_annotations_path': 'ibims/test_annotations.json',
    'test_annotations_path': 'ibims/test_annotations.json',
    'val_annotations_path': 'ibims/test_annotations.json',
    'metric_scale': 1310.7
}

if __name__ == '__main__':
    import os.path as osp
    import json
    dataset_num = input()
    anno_info = mldb_info[dataset_num]

    try:
        train_anno_path = osp.join(anno_info['mldb_root'], anno_info['train_annotations_path'])
        with open(train_anno_path, 'r') as f:
            train_annos = json.load()
        try:
            print('len_train_anno :', len(train_annos['files']))
        except:
            print('len_train_anno :', len(train_annos))
    except:
        pass

    try:
        test_anno_path = osp.join(anno_info['mldb_root'], anno_info['test_annotations_path'])
        with open(test_anno_path, 'r') as f:
            test_annos = json.load()
        try:
            print('len_test_anno :', len(test_annos['files']))
        except:
            print('len_test_anno :', len(test_annos))
    except:
        pass