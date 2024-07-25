import os
import cv2

import numpy as np

from argparse import ArgumentParser

'''
python get_binary_mask.py \
    --input_dir "/test/fcx/codes/marigold/logs/logs_sd21_dis5k/checkpoint-final_eval/DIS-VD/seg_colored" \
    --output_dir "/test/fcx/codes/marigold/logs/logs_sd21_dis5k/checkpoint-final_eval/DIS-VD/seg_colored" \
    --norm_type subdivision \
    --low_pixel_value 50 \
    --high_pixel_value 120
'''

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/test/fcx/codes/marigold/logs/logs_sd21_dis5k/checkpoint-final_eval/DIS-TE4/seg_colored')
    parser.add_argument('--output_dir', type=str, default='/test/fcx/codes/marigold/logs/logs_sd21_dis5k/checkpoint-final_eval/DIS-TE4/seg_colored_subdivision_50_120')
    parser.add_argument('--norm_type', default='subdivision', choices=['maxmin', '255', 'subdivision'])
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--low_pixel_value', type=int, default=50)
    parser.add_argument('--high_pixel_value', type=int, default=100)

    args = parser.parse_args()

    if args.norm_type == 'subdivision':
        args.output_dir = '{}_subdivision_{}_{}'.format(args.output_dir, args.low_pixel_value, args.high_pixel_value)
    else:
       args.output_dir = '{}_{}_{}'.format(args.output_dir, args.norm_type, args.threshold)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    files = os.listdir(args.input_dir)

    for file in files:
        if not file.endswith('.png'):
            continue

        print('>>> processing {}'.format(file))
        in_path = os.path.join(args.input_dir, file)
        out_path = os.path.join(args.output_dir, file)

        # read image, convert to binary mask using threshold
        img = cv2.imread(in_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if args.norm_type == 'maxmin':
            img = (img - img.min()) / (img.max() - img.min())
            img[img >= args.threshold] = 255
            img[img < args.threshold] = 0
        elif args.norm_type == '255':
            img = img / 255.0
            img[img >= args.threshold] = 255
            img[img < args.threshold] = 0
        elif args.norm_type == 'subdivision':
            img[img >= args.high_pixel_value] = 255
            img[img < args.low_pixel_value] = 0

        img = img.astype(np.uint8)

        # save binary mask
        # print('>>> save {}'.format(out_path))
        cv2.imwrite(out_path, img)