import os
import datasets
import pandas as pd
import numpy as np
from numpy import asarray
from io import BytesIO
from PIL import Image
# from diffusers_custom.my_image import MyImage
from datasets.features.image import image_to_bytes
# from pcache_fileio import fileio
# from pcache_fileio import exists
# pip install oss2 pcache-fileio -i https://pypi.antfin-inc.com/simple

_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        # "image": MyImage(),
        # "seg_conditioning_image": MyImage(),
        "image": datasets.Image(),
        "semseg_conditioning_image": datasets.Image(),
        "depth_conditioning_image": datasets.Image(),
        "normal_conditioning_image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)
DATA_DIR = "./datasets"

class Dataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )
 
    
    def _split_generators(self, dl_manager):
        # metadata_path = f"{DATA_DIR}/meta.jsonl"
        # images_dir = f"{DATA_DIR}/images"
        # conditioning_images_dir = f"{DATA_DIR}/processed_images"
        # images_dir = f"{DATA_DIR}/source"
        # conditioning_images_dir = f"{DATA_DIR}/target"

        datasets_list = []
        # for dataset_name in ['coco']:
        for dataset_name in ['hypersim']:
            if dataset_name == 'coco':
                metadata_paths = []
                for i in range(128):
                # for i in range(1):
                    metadata_path = "{}/coco/pano_sem_seg/jsons/coco_train2017_image_panoptic_sem_seg_{}.jsonl".format(DATA_DIR, i)
                    metadata_paths.append(metadata_path)

                images_dir = f"{DATA_DIR}"
                semseg_conditioning_images_dir = f"{DATA_DIR}"
                depth_conditioning_images_dir = f"{DATA_DIR}"

            if dataset_name == 'hypersim':
                metadata_paths = []
                for i in range(128):
                # for i in range(1):
                    metadata_path = "{}/hypersim/jsons/hypersim_train_depth_normal1_semseg_{}.jsonl".format(DATA_DIR, i)
                    metadata_paths.append(metadata_path)

                images_dir = f"{DATA_DIR}"
                semseg_conditioning_images_dir = f"{DATA_DIR}"
                depth_conditioning_images_dir = f"{DATA_DIR}"
                normal_conditioning_images_dir = f"{DATA_DIR}"

            datasets_list.append(
                datasets.SplitGenerator(
                    # name=datasets.Split.TRAIN,
                    name=datasets.splits.NamedSplit(dataset_name),
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "metadata_paths": metadata_paths,
                        "images_dir": images_dir,
                        "depth_conditioning_images_dir": depth_conditioning_images_dir,
                        "normal_conditioning_images_dir": normal_conditioning_images_dir,
                        "semseg_conditioning_images_dir": semseg_conditioning_images_dir,
                        })
            )               
        
        return datasets_list

    
    def _generate_examples(self, 
                           metadata_paths, 
                           images_dir,
                           depth_conditioning_images_dir, 
                           normal_conditioning_images_dir, 
                           semseg_conditioning_images_dir,
                           ):
        for metadata_path in metadata_paths:
            print(metadata_path)
            metadata = pd.read_json(metadata_path, lines=True)
            # if 'laion' in metadata_path:
            #     metadata = metadata.sample(frac=0.005).reset_index(drop=True)
            # elif 'coyo' in metadata_path:
            #     metadata = metadata.sample(frac=0.03).reset_index(drop=True)
            # elif 'bedlam' in metadata_path or 'hi4d' in metadata_path:
            #     metadata = metadata.sample(frac=0.99).reset_index(drop=True)
            # elif 'xhumans' in metadata_path:
            #     metadata = metadata.sample(frac=0.4).reset_index(drop=True)
            # elif 'moyo' in metadata_path:
            #     metadata = metadata.sample(frac=0.06).reset_index(drop=True)
            # elif 'yoga_dance' in metadata_path:
            #     metadata = metadata.sample(frac=0.999).reset_index(drop=True)

            # source: conditional image target: image
            for _, row in metadata.iterrows():
                text = row["caption"]
                # text = row['prompt']
                try:
                    image_path = row["image"]
                    # image_path = row["target"]
                    image_path = os.path.join(images_dir, image_path)                
                    # depth_conditioning_image_path = os.path.join(
                    #     depth_conditioning_images_dir, row["depth_conditioning_image"]
                    # )

                    if "semseg_conditioning_image" in row.keys():
                        semseg_conditioning_image_path = os.path.join(
                            semseg_conditioning_images_dir, row["semseg_conditioning_image"])
                    else:      
                        semseg_conditioning_image_path = None

                    # print(semseg_conditioning_image_path)

                    if "depth_conditioning_image" in row.keys():
                        depth_conditioning_image_path = os.path.join(
                            depth_conditioning_images_dir, row["depth_conditioning_image"])
                    else:      
                        depth_conditioning_image_path = None


                    if "normal_conditioning_image" in row.keys():
                        normal_conditioning_image_path = os.path.join(
                            depth_conditioning_images_dir, row["normal_conditioning_image"])
                    else:      
                        normal_conditioning_image_path = None

                    # text = depth_conditioning_image_path

                    out_dict = {
                        "text": text,
                        "image": {
                            "path": image_path,
                            # "bytes": image,
                        },
                    }

                    if semseg_conditioning_image_path is not None and os.path.isfile(semseg_conditioning_image_path):
                        out_dict["semseg_conditioning_image"] = {
                            "path": semseg_conditioning_image_path,
                            # "bytes": hed_conditioning_image,
                        }
                    else:
                        semseg_conditioning_image = Image.new("RGB", (768, 768), (0,0,0))
                        semseg_conditioning_image = image_to_bytes(semseg_conditioning_image)
                        out_dict["semseg_conditioning_image"] = {
                            # "path": hed_conditioning_image_path,
                            "bytes": semseg_conditioning_image,
                        }


                    if depth_conditioning_image_path is not None and os.path.isfile(depth_conditioning_image_path):
                        out_dict["depth_conditioning_image"] = {
                            "path": depth_conditioning_image_path,
                            # "bytes": hed_conditioning_image,
                        }
                    else:
                        depth_conditioning_image = Image.new("RGB", (768, 768), (0,0,0))
                        depth_conditioning_image = image_to_bytes(depth_conditioning_image)
                        out_dict["depth_conditioning_image"] = {
                            "path": depth_conditioning_image_path,
                            "bytes": depth_conditioning_image,
                        }


                    if normal_conditioning_image_path is not None and os.path.isfile(normal_conditioning_image_path):
                        out_dict["normal_conditioning_image"] = {
                            "path": normal_conditioning_image_path,
                            # "bytes": hed_conditioning_image,
                        }
                    else:
                        normal_conditioning_image = Image.new("RGB", (768, 768), (0,0,0))
                        normal_conditioning_image = image_to_bytes(normal_conditioning_image)
                        out_dict["normal_conditioning_image"] = {
                            "path": normal_conditioning_image_path,
                            "bytes": normal_conditioning_image,
                        }


                    yield row["image"], out_dict
                    
                except Exception as e:
                    print(e)