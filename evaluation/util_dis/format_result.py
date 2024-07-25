import os

import numpy as np
import pickle as pkl

from argparse import ArgumentParser

def format_result(pred_root):
    current_soc_pkl_path = os.path.join(pred_root, "soc_metric.pkl")
    with open(current_soc_pkl_path, "rb") as f:
        current_soc_pkl = pkl.load(f)
    
    current_hce_pkl_path = os.path.join(pred_root, "hce_metric.pkl")
    with open(current_hce_pkl_path, "rb") as f:
        current_hce_pkl = pkl.load(f)
    
    hce = np.mean(np.array(current_hce_pkl['hces'])[:,-1])

    result = {
        'maxFm': current_soc_pkl['maxFm'],
        'wFmeasure': current_soc_pkl['wFmeasure'],
        'MAE': current_soc_pkl['MAE'],
        'Smeasure': current_soc_pkl['Smeasure'],
        'meanEm': current_soc_pkl['meanEm'],
        'hce': hce,
    }

    return result

def print_result(result):
    print(result['maxFm'].round(3))
    print(result['wFmeasure'].round(3))
    print(result['MAE'].round(3))
    print(result['Smeasure'].round(3))
    print(result['meanEm'].round(3))
    print('{:.0f}'.format(result['hce']))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred_root', type=str, default='/test/fcx/codes/marigold/logs/logs_sd21_dis5k/checkpoint-final_eval/')
    parser.add_argument('--group', type=str, default='seg_colored_maxmin_0.6')
    args = parser.parse_args()

    print('pred_root: {}'.format(args.pred_root))

    dataset = ["DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4"]

    final_result = {}

    # validation
    result = format_result(os.path.join(args.pred_root, "DIS-VD", args.group))
    final_result["DIS-VD"] = result
    print_result(result)

    # test
    for d in dataset:
        result = format_result(os.path.join(args.pred_root, d, args.group))
        final_result[d] = result
        print_result(result)
    
    overall_result = {
        'maxFm': 0,
        'wFmeasure': 0,
        'MAE': 0,
        'Smeasure': 0,
        'meanEm': 0,
        'hce': 0,
    }

    for d in dataset:
        overall_result['maxFm'] += final_result[d]['maxFm']
        overall_result['wFmeasure'] += final_result[d]['wFmeasure']
        overall_result['MAE'] += final_result[d]['MAE']
        overall_result['Smeasure'] += final_result[d]['Smeasure']
        overall_result['meanEm'] += final_result[d]['meanEm']
        overall_result['hce'] += final_result[d]['hce']
    
    overall_result['maxFm'] /= len(dataset)
    overall_result['wFmeasure'] /= len(dataset)
    overall_result['MAE'] /= len(dataset)
    overall_result['Smeasure'] /= len(dataset)
    overall_result['meanEm'] /= len(dataset)
    overall_result['hce'] /= len(dataset)

    final_result['overall'] = overall_result
    print_result(overall_result)

    with open(os.path.join(args.pred_root, "final_result.pkl"), "wb") as f:
        pkl.dump(final_result, f)