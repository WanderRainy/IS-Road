#!/user/bin/python
# coding=utf-8
# 修改、来源：VecRoad github
# imageid列中需为AOI_0_amsterdam_img0格式
import os
import argparse
from multiprocessing import Pool


def worker(fn,args):
    fn = fn.split('.')[0]
    ret = os.popen(
        "java -jar {} -truth {}/{}.csv -solution {}/{}.csv -no-gui"
        .format(args.apls_path, args.gt_dir, fn, args.wkt_dir, fn)
    ).readlines()
    print(ret)
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_name", type=str, help="file_name to name result.csv", default="exp9_1_v2"
    )
    parser.add_argument(
        "--wkt_dir", type=str, help="input predict wkt dir", default=r"C:\Users\Rain\Desktop\centerline\RoadSegment\results_city\exp9_1_v2"#C:\Users\Rain\Desktop\centerline\RoadSegment\results_city\exp7_1_large_csv"
    )
    parser.add_argument(
        "--gt_dir", type=str, help="input gt wkt dir", default=r"C:\Users\Rain\Desktop\centerline\RoadSegment\city_scale\20cities_test_wkt"#C:\Users\Rain\Desktop\centerline\data_baseline\RGB_1.0_meter_wkt"#C:\Users\Rain\Desktop\centerline\RoadSegment\city_scale\20cities_test_wkt"
    ) # C:\Users\Rain\Desktop\centerline\RoadSegment\city_scale\20cities_patch\graph_wkt_test
    parser.add_argument(
        "--save_dir", type=str, help="save result.csv dir", default=r"C:\Users\Rain\Desktop\centerline\RoadSegment\results_city"
    )
    parser.add_argument(
        "--apls_path", type=str, help="apls metric jar dir", default=r"C:\Users\Rain\Desktop\centerline\eval_metric\apls-visualizer-1.0\visualizer.jar"
    )

    args = parser.parse_args()

    files = os.listdir(args.gt_dir)
    files.sort()

    res_lst = []
    # files = ['denver.graph']
    pool = Pool()
    tmp_lst = []
    ret_lst = []
    for fn in files:
        tmp_lst.append(pool.apply_async(worker, args=(fn,args)))
        # continue
    for item in tmp_lst:
        ret_lst.append(item.get())
    pool.close()
    pool.join()

    res_file = open(os.path.join(args.save_dir, '{}.csv'.format(args.file_name)), 'w')
    log_file = open(os.path.join(args.save_dir, '{}.log'.format(args.file_name)), 'w')
    for ret in ret_lst:
        for line in ret:
            log_file.write(line)
        log_file.write('\n')
        try:
            res_lst.append(float(ret[-1].strip().split(' ')[-1]))
        except:
            if ret[2] =='Nothing to score\n':
                res_lst.append(None) # gt为空
            elif ret[-3]=='Overall score : 0\n':
                res_lst.append(0)
            elif ret[1] =='Error reading roads\n':
                res_lst.append(None) # solution为空，不确定gt情况
            else:
                print(ret)
        #     res_lst.append(0.0)
    for i in range(len(files)):
        fn = files[i]
        res = res_lst[i]
        res_file.write("{},{}\n".format(fn, res))
    res_file.write("{},{}".format(args.file_name, sum(list(filter(None,res_lst)))/len(list(filter(None,res_lst)))))
    res_file.close()
    log_file.close()
