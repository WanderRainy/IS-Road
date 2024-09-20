#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 修改在vecroad 的eval_pixel的基础上
import argparse
import glob
import os
import sys
from multiprocessing import Pool
from os.path import basename

import numpy as np
import numba
import cv2 as cv
import pandas as pd

parser = argparse.ArgumentParser()
# parser.add_argument('--gt_dir', type=str, default=r"C:\Users\Rain\Desktop\centerline\data_baseline\RGB_1.0_meter_wkt")
parser.add_argument('--gt_dir', type=str, default=r"C:\Users\Rain\Desktop\centerline\RoadSegment\city_scale\20cities_test_wkt")
# C:\Users\Rain\Desktop\centerline\RoadSegment\city_scale\20cities_patch\graph_wkt_test
#C:\Users\Rain\Desktop\centerline\data_baseline\RGB_1.0_meter_wkt
#C:\Users\Rain\Desktop\centerline\RoadSegment\city_scale\20cities_test_wkt
# parser.add_argument('--pred_dir', type=str, default=r"C:\Users\Rain\Desktop\centerline\RoadSegment\results\exp8_3") #C:\Users\Rain\Desktop\centerline\RoadSegment\results_city\exp7_1_large_csv
# City-scale
## DinkNet
# parser.add_argument('--pred_dir', type=str, default=r"C:\Users\Rain\Desktop\centerline\GAMSNet\DLinkNet_city_wkt")
# parser.add_argument('--save_metric_path', type=str, default=r"C:\Users\Rain\Desktop\centerline\GAMSNet\DLinkNet_city_pixel_metric.txt")
## GAMSNet
# parser.add_argument('--pred_dir', type=str, default=r"C:\Users\Rain\Desktop\centerline\GAMSNet\GAMSNet_city_wkt")
# parser.add_argument('--save_metric_path', type=str, default=r"C:\Users\Rain\Desktop\centerline\GAMSNet\GAMSNet_city_wkt_csv_metric.txt")
## RNGDet
# parser.add_argument('--pred_dir', type=str, default=r"C:\Users\Rain\Desktop\centerline\RNGDetPlusPlus\cityscale\results\wkt_rngdet_author")
# parser.add_argument('--save_metric_path', type=str, default=r"C:\Users\Rain\Desktop\centerline\RNGDetPlusPlus\cityscale\results\wkt_rngdet_author_csv_metric.txt")
## Sat2graph
# parser.add_argument('--pred_dir', type=str, default=r"C:\Users\Rain\Desktop\centerline\Sat2Graph\outputs_wkt")
# parser.add_argument('--save_metric_path', type=str, default=r"C:\Users\Rain\Desktop\centerline\Sat2Graph\city_scale_outputs_sn_p2wkt_pixel_metric.txt")
## IS-RoadDet
parser.add_argument('--pred_dir', type=str, default=r"C:\Users\Rain\Desktop\centerline\RoadSegment\results_city\exp9_4_v2")
parser.add_argument('--save_metric_path', type=str, default=r"C:\Users\Rain\Desktop\centerline\RoadSegment\results_city\exp9_4_v2_metric.txt")
parser.add_argument('--relax', type=int, default=3.5, help='radius')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()
print(args)

result_dir = args.pred_dir
label_dir = args.gt_dir
# label_fns = sorted(glob.glob('%s/*.csv' % label_dir))
label_fns = sorted(glob.glob('%s/region_128.csv' % label_dir))
n_results = len(label_fns)


def get_pre_rec(positive, prec_tp, true, recall_tp):
    pre_rec = []
    breakeven = []

    t = 0
    if positive[t] < prec_tp[t] or true[t] < recall_tp[t]:
        sys.exit('calculation is wrong')
    pre = float(prec_tp[t]) / positive[t] if positive[t] > 0 else 0
    rec = float(recall_tp[t]) / true[t] if true[t] > 0 else 0
    pre_rec.append([pre, rec])
    if pre != 1 and rec != 1 and pre > 0 and rec > 0:
        breakeven.append([pre, rec])

    pre_rec = np.asarray(pre_rec)

    breakeven = np.asarray(breakeven)
    breakeven_pt = np.abs(breakeven[:, 0] - breakeven[:, 1]).argmin()
    breakeven_pt = breakeven[breakeven_pt]

    return pre_rec, breakeven_pt

def wkt2seg(csvpath):
    """
    vecroad代码中pixel指标的计算是基于栅格的，所以需要将wkt转为栅格上
    :param csvpath: ***.csv
    :return:
    """
    wkts = pd.read_csv(csvpath)["WKT_Pix"].tolist()
    # 从wkt 到线集的表示
    lines = []
    for wkt in wkts:
        line=[]
        line_str = wkt[12:-1].split(',')
        for point in line_str:
            line.append([int(float(x)) for x in point.split(' ')[-2:]])
        lines.append(line)
    if lines == []:
        return np.zeros((10,10),np.uint8) # 有的csv为空，返回全0数组即可
    max_number = max([number for line in lines for point in line for number in point]) # 获取最大值，用于栅格化的栅格大小

    seg = np.zeros((max_number, max_number), np.uint8)
    for line in lines:
        cv.polylines(seg, [np.array(line).reshape((-1, 1, 2))], False, 1)

    return seg

def worker(img_idx, result_fn):
    img_id = basename(result_fn).split('.')[0]

    label = wkt2seg('%s/%s.csv' % (label_dir, img_id))
    if os.path.exists('%s/%s.csv' % (result_dir, img_id)):
        pred = wkt2seg('%s/%s.csv' % (result_dir, img_id))
    else:
        pred = np.zeros((10,10),np.uint8)
    # 画线时的栅格由于是用最大点判断的的，因此可能pred和label不一样大，统一一下
    size = max(label.shape[0],pred.shape[0])
    label = np.pad(label, ((0,size-label.shape[0]),(0,size-label.shape[0])))
    pred = np.pad(pred, ((0, size - pred.shape[0]),(0, size - pred.shape[0])))

    positive = []
    prec_tp = []
    true = []
    recall_tp = []
    pred_vals = np.array(pred, dtype=np.int32)
    label_vals = np.array(label, dtype=np.int32)

    positive.append(np.sum(pred_vals))
    prec_tp.append(relax_precision(pred_vals, label_vals, args.relax))
    true.append(np.sum(label_vals))
    recall_tp.append(relax_recall(pred_vals, label_vals, args.relax))

    print('thread finished')
    return positive, prec_tp, true, recall_tp


@numba.jit
def relax_precision(predict, label, relax):
    h_lim = predict.shape[1]
    w_lim = predict.shape[0]

    true_positive = 0

    for y in range(h_lim):
        for x in range(w_lim):
            if predict[y, x] == 1:
                st_y = y - relax if y - relax >= 0 else 0
                en_y = y + relax if y + relax < h_lim else h_lim - 1
                st_x = x - relax if x - relax >= 0 else 0
                en_x = x + relax if x + relax < w_lim else w_lim - 1

                sum = 0
                for yy in range(st_y, en_y + 1):
                    for xx in range(st_x, en_x + 1):
                        sum += label[yy, xx]
                if sum > 0:
                    true_positive += 1

    return true_positive


@numba.jit
def relax_recall(predict, label, relax):
    h_lim = predict.shape[1]
    w_lim = predict.shape[0]

    true_positive = 0

    for y in range(h_lim):
        for x in range(w_lim):
            if label[y, x] == 1:
                st_y = y - relax if y - relax >= 0 else 0
                en_y = y + relax if y + relax < h_lim else h_lim - 1
                st_x = x - relax if x - relax >= 0 else 0
                en_x = x + relax if x + relax < w_lim else w_lim - 1

                sum = 0
                for yy in range(st_y, en_y + 1):
                    for xx in range(st_x, en_x + 1):
                        sum += predict[yy, xx]
                if sum > 0:
                    true_positive += 1

    return true_positive


def get_f1(pre, rec):
    if pre == rec == 0:
        return 0
    return 2 * pre * rec / (pre + rec)


if __name__ == '__main__':
    pool = Pool(args.num_workers)
    tmp_lst = []
    for i in range(n_results):
        tmp_lst.append(pool.apply_async(worker, args=(i, label_fns[i],)))
    for i in range(n_results):
        tmp_lst[i] = tmp_lst[i].get()
    print('all finished')

    all_positive = np.array([x[0] for x in tmp_lst])
    all_prec_tp = np.array([x[1] for x in tmp_lst])
    all_true = np.array([x[2] for x in tmp_lst])
    all_recall_tp = np.array([x[3] for x in tmp_lst])

    all_positive = np.sum(all_positive, axis=0)
    all_prec_tp = np.sum(all_prec_tp, axis=0)
    all_true = np.sum(all_true, axis=0)
    all_recall_tp = np.sum(all_recall_tp, axis=0)

    pre_rec, breakeven_pt = get_pre_rec(
        all_positive, all_prec_tp,
        all_true, all_recall_tp)
    F1 = []
    # for i in range(args.steps):
    if pre_rec[0, 0] != 0.:
        # continue
        F1.append(get_f1(pre_rec[0, 0], pre_rec[0, 1]))

    # print("BreakEven Point:")
    print("precision: {:.2f} Recall: {:.2f} F1: {:.2f}".format(
        breakeven_pt[0] * 100, breakeven_pt[1] * 100, get_f1(*breakeven_pt) * 100))
    with open(args.save_metric_path,'a') as file:
        file.write("only R128")
        file.write("Relax:{} precision: {:.2f} Recall: {:.2f} F1: {:.2f}\n".format(args.relax,
        breakeven_pt[0] * 100, breakeven_pt[1] * 100, get_f1(*breakeven_pt) * 100))
    # print("max P-F1: {:.2f}".format(np.max(F1) * 100))
