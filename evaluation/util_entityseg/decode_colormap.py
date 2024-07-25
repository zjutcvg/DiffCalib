
import os
os.environ["OPENBLAS_NUM_THREADS"] = "128"

import cv2
import os
import time
import copy
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

from argparse import ArgumentParser
from sklearn.cluster import KMeans
from pycocotools import mask as maskUtils

'''
python evaluation/util_entityseg/decode_colormap.py \
    --input_path /test/fcx/codes/marigold/ccc.png \
    --threshold 10.0 \
    --out_dir /test/fcx/codes/marigold/delta_10/ccc \
    --save_mask
'''

def progressive_dichotomy_module(img, mask, delta, mask_list):
    # Calulate the l2 distance of the mask
    h, w, c = img.shape
    mean = np.mean(img[mask == 1], axis=0)
    dist = np.sum((img[mask == 1] - mean)**2, axis=0)
    dist = dist / (h * w)

    if (dist < delta).all():
        mask_list.append(mask)
        return
    
    # Split mask into two sub-masks using two-cluster k-means
    mask1, mask2 = two_cluster_kmeans(img, mask)
    
    # Recursive call for further splitting
    progressive_dichotomy_module(img, mask1, delta, mask_list)
    progressive_dichotomy_module(img, mask2, delta, mask_list)

def two_cluster_kmeans(img, mask):
    # Apply two-cluster k-means to split the mask into two sub-masks based on img features
    
    # Flatten the img for pixels in the mask
    img_flat = img[mask == 1].reshape(-1, img.shape[-1])
    
    # Apply k-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(img_flat)
    
    # Map labels back to the original mask shape
    mask1 = np.zeros_like(mask)
    mask2 = np.zeros_like(mask)
    mask1[mask == 1] = labels
    mask2[mask == 1] = 1 - labels

    return mask1, mask2

def colormap_decoding(input_path, image_id=0, args=None):
    # Read the input image
    filename = os.path.splitext(os.path.basename(input_path))[0]
    image = cv2.imread(input_path)
    
    # Convert BGR to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Concatenate BGR and LAB
    image = np.concatenate((image, lab_image), axis=2)
    
    # Apply progressive dichotomy module
    mask_reuslt = []
    init_mask = np.ones_like(image[:, :, 0])
    progressive_dichotomy_module(image, init_mask, args.threshold, mask_reuslt)
    
    # Save the resulting mask
    if args.save_mask:
        if not os.path.exists(args.out_dir):
            print('>> create {}'.format(args.out_dir))
            os.makedirs(args.out_dir)
    
        for i, mask in enumerate(mask_reuslt):
            out_path = '{}/{}_m{}.png'.format(args.out_dir, filename, i)
            print('>>> save mask to {}'.format(out_path))
    
            mask = mask.astype(np.uint8) * 255
    
            cv2.imwrite(out_path, mask)

    # # Concatenate original img and the masks using matplotlib

    # fig = plt.figure(figsize=(20, 40))
    # gs = gridspec.GridSpec(1, len(mask_reuslt) + 1)
    # gs.update(wspace=0.025, hspace=0.05)

    # ax = plt.subplot(gs[0])
    # ax.imshow(image[:, :, :3])
    # ax.axis('off')
    # ax.set_title('ori', fontsize=20)

    # for i, mask in enumerate(mask_reuslt):
    #     ax = plt.subplot(gs[i + 1])
    #     ax.imshow(mask, cmap='gray')
    #     ax.axis('off')
    #     ax.set_title('mask {}'.format(i + 1), fontsize=20)
    
    # # plt.show()
    # plt.savefig('{}/{}_result.png'.format(out_dir, filename))

    # save to coco json
    if args.save_json:
        output_coco_json = []
    
        for i, mask in enumerate(mask_reuslt):
            mask = mask.astype(np.uint8) * 255
            mask = np.asfortranarray(mask)
            rle = maskUtils.encode(mask)
            bbox = maskUtils.toBbox(rle)
            rle['counts'] = rle['counts'].decode('utf-8')
            output_coco_json.append({
                'image_id': image_id,
                'category_id': 1,
                'bbox': bbox.tolist(),
                'score': 1,
                'segmentation': rle
            })
        
        return output_coco_json
    
if __name__ == '__main__':
    parser = ArgumentParser(description='Colormap Decoding')
    parser.add_argument('--input_path', type=str, default='20240106-143151.jpg', help='Input image path')
    parser.add_argument('--in_dir', type=str, default=None, help='Input image path')
    parser.add_argument('--threshold', type=float, default=10, help='Threshold for stopping criteria')
    parser.add_argument('--out_dir', type=str, default='output/delta_10', help='Output directory for masks')
    parser.add_argument('--save_json', default=False, action='store_true', help='Save coco json')
    parser.add_argument('--save_mask', default=False, action='store_true', help='Save mask')
    
    args = parser.parse_args()

    start_time = time.time()
    if args.in_dir != None:
        if args.save_json:
            out_json = []

        for file in os.listdir(args.in_dir):
            if not file.endswith('.jpg') \
                and not file.endswith('.png') \
                and not file.endswith('.jpeg'):
                continue
            input_path = os.path.join(args.in_dir, file)
            print('>>> processing {}'.format(input_path))
            image_id = 0

            if args.save_json:
                result = colormap_decoding(input_path, image_id, args)
                out_json.extend(result)
            else:
                colormap_decoding(input_path, image_id, args)
        
        if args.save_json:
            with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
                json.dump(out_json, f)

    else:
        colormap_decoding(args.input_path, args=args)
    end_time = time.time()

    print('>>> total time: {:.2f}s'.format(end_time - start_time))

del os.environ["OPENBLAS_NUM_THREADS"]