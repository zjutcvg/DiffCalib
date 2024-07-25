#!/bin/bash
pred_root="/test/fcx/codes/marigold/logs/logs_sd21_dis5k/checkpoint-final_eval/"
gt_root="/test/fcx/codes/marigold/datasets/dis5k/DIS5K/"
gt_ske_root="/test/fcx/codes/marigold/datasets/dis5k/skeletion/"
group="seg_colored_maxmin_0.6"
dataset=("DIS-VD" "DIS-TE1" "DIS-TE2" "DIS-TE3" "DIS-TE4")

# remove last '/' in pred_root, gt_root and gt_ske_root if exists
pred_root=${pred_root%/}
gt_root=${gt_root%/}
gt_ske_root=${gt_ske_root%/}

for i in "${dataset[@]}"
do
    echo "Processing SOC $i"
    current_pred_root=$pred_root/$i/$group
    current_gt_root=$gt_root/$i/gt
    python soc_eval.py --pred_root "$current_pred_root" --gt_root "$current_gt_root" &
done

for i in "${dataset[@]}"
do
    echo "Processing HCE $i"
    current_pred_root=$pred_root/$i/$group
    current_gt_root=$gt_root/$i/gt
    current_gt_ske_root=$gt_ske_root/$i

    # if i == dataset[0] single process, else multi-process
    if [ "$i" == "${dataset[0]}" ]; then
        python hce_metric_main.py --pred_root "$current_pred_root" --gt_root "$current_gt_root" --gt_ske_root "$current_gt_ske_root"
    else
        python hce_metric_main.py --pred_root "$current_pred_root" --gt_root "$current_gt_root" --gt_ske_root "$current_gt_ske_root" && echo "Done $i" &
    fi
done