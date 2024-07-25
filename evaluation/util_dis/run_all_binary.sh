#!/bin/bash
pred_root="/test/fcx/codes/marigold/logs/logs_sd21_dis5k/checkpoint-final_eval/"
norm_type="maxmin"
threshold=0.6
# low_pixel_value=40
# high_pixel_value=200
dataset=("DIS-VD" "DIS-TE1" "DIS-TE2" "DIS-TE3" "DIS-TE4")

# remove last '/' in pred_root, gt_root and gt_ske_root if exists
pred_root=${pred_root%/}

# subdivision
# for i in "${dataset[@]}"
# do
#     echo "Binary $i"
#     current_pred_root=$pred_root/$i/seg_colored
#     python get_binary_mask.py --input_dir "$current_pred_root" --output_dir "$current_pred_root" --norm_type $norm_type --low_pixel_value $low_pixel_value --high_pixel_value $high_pixel_value && echo "Done $i" &
# done

# maxmin
for i in "${dataset[@]}"
do
    echo "Binary $i"
    current_pred_root=$pred_root/$i/seg_colored
    python get_binary_mask.py --input_dir "$current_pred_root" --output_dir "$current_pred_root" --norm_type $norm_type --threshold $threshold && echo "Done $i" &
done