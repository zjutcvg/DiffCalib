import os
import sys
import cv2
from tqdm import tqdm
import metrics as M
import json
import argparse
import pickle as pkl

'''
python soc_eval.py \
    --pred_root "/test/fcx/codes/marigold/logs/logs_sd21_dis5k/checkpoint-final_eval/DIS-VD/seg_colored" \
    --gt_root "/test/fcx/codes/marigold/datasets/dis5k/DIS5K/DIS-VD/gt"
'''

def main():
    FM = M.Fmeasure()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()

    gt_root = args.gt_root
    pred_root = args.pred_root

    gt_name_list = sorted(os.listdir(pred_root))

    for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
        if not gt_name.endswith('.png'):
            continue

        #print(gt_name)
        gt_path = os.path.join(gt_root, gt_name.replace('_seg_pred_steps_51_colored', ''))
        pred_path = os.path.join(pred_root, gt_name)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)

    fm = FM.get_results()['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']

    soc_metric = {
        'fm': fm,
        'em': em,
        'Smeasure': sm,
        'wFmeasure': wfm,
        'MAE': mae,
        'adpEm': em['adp'],
        'meanEm': '-' if em['curve'] is None else em['curve'].mean(),
        'maxEm': '-' if em['curve'] is None else em['curve'].max(),
        'adpFm': fm['adp'],
        'meanFm': fm['curve'].mean(),
        'maxFm': fm['curve'].max(),
    }

    file_metric = open(pred_root + '/soc_metric.pkl','wb')
    pkl.dump(soc_metric, file_metric)
    # file_metrics.write(cmn_metrics)
    file_metric.close()

    print(
        'maxFm:', fm['curve'].max().round(3), '; ',
        'wFmeasure:', wfm.round(3), '; ',
        'MAE:', mae.round(3), '; ',
        'Smeasure:', sm.round(3), '; ',
        'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
        'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(3), '; ',
        'adpEm:', em['adp'].round(3), '; ',
        'adpFm:', fm['adp'].round(3), '; ',
        'meanFm:', fm['curve'].mean().round(3), '; ',
        sep=''
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_root', type=str, default='/home/fcx/codes/DIS/output/sotas-dis-eccv2022/isnet(ours)/DIS-TE4')
    parser.add_argument('--gt_root', type=str, default='/mnt/nas/share2/home/xugk/marigold_github_repo_data_zips_temp/DIS5K/DIS-TE4/gt')
    args = parser.parse_args()

    main()
