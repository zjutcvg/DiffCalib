from PIL import Image
import argparse
import os
import tqdm
import shutil
import numpy as np
import cv2

def generate_color_values():
    # colors = [0, 64, 128, 192, 255]
    #colors = [0, 32, 64, 96, 128, 160, 192, 224, 255]
    # [8, 24, 40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 200, 216, 232, 248]
    colors = list(range(8, 256, 16))
    color_values = []
    for r in colors:
        for g in colors:
            for b in colors:
                color_values.append([r, g, b])
#     return color_values[100:]  # 去掉前面种颜色
#     #return color_values[29:]  # 去掉前面种颜色
    return color_values

def generate_color_values_unigs():
    colors = [0, 64, 128, 192, 255]
    color_values = []
    for r in colors:
        for g in colors:
            for b in colors:
                if r == 0 \
                    and g == 0 \
                    and b == 0:
                    continue
                color_values.append([r, g, b])
    
    print('>>> color_values len: {}'.format(len(color_values)))
    print('>>> color_values: {}'.format(color_values))
    return color_values

def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8))
    lbl_pil.save(save_path)
    # print('>>> save mask to {}'.format(save_path))
    # imageio.imwrite(save_path, mask.astype(np.uint16))

def assign_color_to_bbox_center(bbox, image_width, image_height, num_grid_x, num_grid_y, color_values=None):
    # 计算bbox中心点坐标
    x, y, width, height = bbox
    center_x = x + width / 2
    center_y = y + height / 2

    # 计算中心点在11x11区域中的位置
    region_width = image_width / num_grid_x  ##
    region_height = image_height / num_grid_y  ##
    region_x = int(center_x / region_width)
    region_y = int(center_y / region_height)

    assert region_x < num_grid_x, 'region_x {} out of range'.format(region_x)
    assert region_y < num_grid_y, 'region_y {} out of range'.format(region_y)

    # if region_x >= num_grid_x:
    #     print('region_width: {}, region_height: {}'.format(region_width, region_height))
    #     print('center_x: {}, center_y: {}'.format(center_x, center_y))
    #     print('>>> region_x {} out of range'.format(region_x))
    
    # if region_y >= num_grid_y:
    #     print('region_width: {}, region_height: {}'.format(region_width, region_height))
    #     print('center_x: {}, center_y: {}'.format(center_x, center_y))
    #     print('>>> region_y {} out of range'.format(region_y))

    # print('>>> region_x: {}, region_y: {}'.format(region_x, region_y))

    # 根据区域位置确定颜色值
    # color_index = (region_x + region_y*22) % len(color_values)
    # color = color_values[int(color_index)]
    # print(color_index)
    color = region_x + region_y * num_grid_x

    return color

def main(args):
    # 定义颜色值列表
    color_values = generate_color_values_unigs()
    
    out_mask_dir = args.output_dir
    if not os.path.exists(out_mask_dir):
        os.makedirs(out_mask_dir)

    in_files = os.listdir(args.input_dir)

    for in_file in in_files:
        current_in_dir = os.path.join(args.input_dir, in_file)

        in_masks = os.listdir(current_in_dir)

        out_mask = None
        img_width = -1
        img_height = -1

        for i, in_mask in enumerate(in_masks):
            current_in_mask_path = os.path.join(current_in_dir, in_mask)
            current_mask = cv2.imread(current_in_mask_path, cv2.IMREAD_GRAYSCALE)

            if i == 0:
                img_width = current_mask.shape[1]
                img_height = current_mask.shape[0]
                out_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

            # get bbox of binary current mask using numpy
            hor = np.sum(current_mask, axis=0)
            hor_idx = np.nonzero(hor)[0]
            x = hor_idx[0]
            width = hor_idx[-1] - x + 1
            vert = np.sum(current_mask, axis=1)
            vert_idx = np.nonzero(vert)[0]
            y = vert_idx[0]
            height = vert_idx[-1] - y + 1
            bbox = [int(x), int(y), int(width), int(height)]
            
            color = assign_color_to_bbox_center(bbox, 
                                                image_width=img_width, 
                                                image_height=img_height, 
                                                num_grid_x=args.num_grid_x, 
                                                num_grid_y=args.num_grid_y)
            out_mask[current_mask==255] = color_values[color]

        current_out_mask_path = os.path.join(out_mask_dir, '{}.png'.format(in_file))
        save_colored_mask(out_mask, current_out_mask_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/test/fcx/codes/marigold/delta_10", type=str,
                        help="input dataset directory")
    parser.add_argument("--output_dir", default="/test/fcx/codes/marigold/delta_10_re", type=str,
                        help="output dataset directory")
    parser.add_argument("--in_json_dir", default="/mnt/nas/datasets2/entityseg/annotations/entityseg_train_01.json", type=str,
                        help="input dataset directory")
    parser.add_argument("--num_grid_x", default=11, type=int,
                        help="num grid x")
    parser.add_argument("--num_grid_y", default=11, type=int,
                        help="num grid y")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)