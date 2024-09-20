
# 输入一个城市的语义分割得到的点概率图和道路段实例掩膜，输出城市graph
# 点概率图格式：'.tif',[0,1],float
## 遍历城市所有点概率图中获得关键点列表
## while 每个道路段：
##      在关键点列表中找到道路段掩膜获得覆盖的关键点,不包含两个及以上关键点的道路段舍去
##      每个道路段掩膜中获得转折点
#       组合所有关键点和转折点形成点集
# while 每个道路段
##      在关键点列表中找到道路段掩膜获得覆盖的关键点,不包含两个及以上关键点的道路段舍去
##      旅行商求解连接成线
##      形成wkt格式道路段
## 保存成csv
import json
import os
# from osgeo import gdal
import tifffile
import elkai
from tqdm import tqdm
import scipy
import numpy as np
import pandas as pd
from pycocotools import mask as maskUtils
import skimage
import skimage.io as io
from skimage.morphology import skeletonize
from utils import sknw
from utils.graph_utils import G2point
from scipy.spatial.distance import euclidean
import networkx as nx

if __name__ == '__main__':
    point_tif_dir = r'C:\Users\Rain\Desktop\centerline\RoadSegment\results_point\exp9_1'  # 语义分割得到的关键点概率图文件夹
    segm_json_path = r"C:\Users\Rain\Desktop\centerline\RoadSegment\results_city\test.segm9_1.json"
    test_json_path = r'city_scale/20cities_patch/Instance_road/test/instances_road_test.json'  # segm中的imgid和图片名关系
    csv_out_dir = 'results_city/exp9_1_v2'
    os.makedirs(csv_out_dir, exist_ok=True)

    # 获取图片名对应的imgid
    test_json = json.load(open(test_json_path, 'rb'))
    id_name = {}
    for image_info in test_json['images']:
        # id_name[image_info['file_name'][:-4]] = image_info['id']
        id_name[image_info['id']] = image_info['file_name'][:-4]#"id": 4, "file_name": "00803.png"

    # 读取测试集
    # testnames = json.load(open(r'data/dataset.json', 'rb'))['train']
    split_dict = json.load(open(r'city_scale/data_split.json', 'r'))
    testidlist = split_dict['test']
    testlist = []
    for r in range(0,7):
        for c in range(0,7):
            testlist += [f'region_{tile_index}_{str(r) + str(c)}' for tile_index in testidlist]
    testnames = testlist

    # 读取所有实例
    segm_json = json.load(open(segm_json_path, 'rb'))

    for testid in tqdm(testidlist):
        # if testid != 49: # 调试时查看一个图片
        #     continue
        point_large = [] # 记录一个region的所有交叉点
        point_connected = [] # 记录已经包含连通性的边
        segments = [] # region中所有的边
        for testname in testnames:
            if os.path.basename(testname).split('_')[-2] == str(testid):# 找到一个region的所有testname
                # 找到一个patch上的点,并合并到region点集合中
                point_tif_path = os.path.join(point_tif_dir, os.path.basename(testname).split('_')[-2].zfill(3) +
                                              os.path.basename(testname).split('_')[-1] + '.tif')
                point_tif = np.array(io.imread(point_tif_path))
                neighborhood = scipy.ndimage.generate_binary_structure(2, 2)
                local_max = (scipy.ndimage.maximum_filter(point_tif, footprint=neighborhood) == point_tif)
                point_tif = local_max & (point_tif > (np.max(point_tif) + np.min(point_tif)) / 2)
                point_coor = np.where(point_tif)
                point_coor = np.vstack((point_coor[0], point_coor[1])).T
                point_coor = [list(x) for x in point_coor]
                ## 根据testname上的行列号，转化patch坐标到region坐标上
                row_id = int(os.path.basename(testname).split('_')[-1])//10
                col_id = int(os.path.basename(testname).split('_')[-1])%10
                ## 第一个patch有，第二个patch对应点舍弃
                for point_single in point_coor:
                    point_single = [point_single[0]+row_id*256,point_single[1]+col_id*256]
                    if point_large == []:# region第一次循环为空
                        point_large += [point_single]
                    else:
                        duplicate_condition = (
                                    np.sqrt(np.sum((np.array(point_single) - np.array(point_large)) ** 2, axis=1)) > 1)
                        if np.all(duplicate_condition): # 没有重复点
                            point_large += [point_single]
                        else: # 已有该点，舍去
                            pass
        ### 得到point_large 一个region上多有交叉点集合

        # for testname in testnames:
        for tempindex, road_instance in enumerate(segm_json):
            if (int(id_name[road_instance['image_id']])//100 == testid) & (road_instance['score'] > 0.75): # 找到region区域的所有instance
                row_id = int(id_name[road_instance['image_id']][-2])
                col_id = int(id_name[road_instance['image_id']][-1])
                mask_large = np.zeros((2048, 2048))
                mask = maskUtils.decode(road_instance['segmentation'])
                mask_dilate = skimage.morphology.dilation(mask, np.ones((10, 10)))
                mask_large[row_id*256:row_id*256+512, col_id*256:col_id*256+512] = mask_dilate
                segment = []  # 该实例道路段中存在的点
                for point in point_large:
                #     # try:
                #     # if point[1] == 512:
                #     #     point[1] = 511
                #     # if point[0] == 512:
                #     #     point[0] = 511
                    if mask_large[point[0], point[1]] == 1:  # graph 中是横轴，纵轴
                        segment.append([int(x) for x in point])
                # if len(segment)>=2:
                    ## 由细化掩膜得到线段
                ske = skeletonize(np.ascontiguousarray(mask_large)).astype(np.uint16)
                G = sknw.build_sknw(ske, multi=True)
                seg_point = G2point(G)## [array[line1]，array[[point1],[point2]]]
                seg_line_points = []##细化得到的所有关键点
                for seg_line in seg_point:
                    seg_line_points.extend(seg_line)

                for point_index, seg_point_i in enumerate(seg_line_points):# 找到已有点图里是否有重复点
                    duplicate_condition = (
                                np.sqrt(np.sum((np.array(seg_point_i) - np.array(point_large)) ** 2, axis=1)) > 1)
                    if np.all(duplicate_condition):
                        # 没有重复点，直接添加点到点集中
                        point_large = point_large + [seg_point_i]
                        # 形成该掩膜内覆盖到的所有点
                        segment = segment + [seg_point_i]
                    else:
                        # 有重复点，用点集中点更新道路段得到地点
                        duplicate_point_index = np.where(duplicate_condition == 0)[0][0]
                        for i, seg_point1 in enumerate(seg_point):
                            for j, seg_point2 in enumerate(seg_point1):
                                if seg_point2 == seg_point_i:
                                    seg_point[i][j] = point_large[duplicate_point_index]
                                    break
                        seg_line_points[point_index] = point_large[duplicate_point_index]
        ## 完成关键点集合的构建，连接点
        for tempindex, road_instance in enumerate(segm_json):
            if (int(id_name[road_instance['image_id']]) // 100 == testid) & (
                    road_instance['score'] > 0.75):  # 找到region区域的所有instance
                row_id = int(id_name[road_instance['image_id']][-2])
                col_id = int(id_name[road_instance['image_id']][-1])
                mask_large = np.zeros((2048, 2048))
                mask = maskUtils.decode(road_instance['segmentation'])
                mask_dilate = skimage.morphology.dilation(mask, np.ones((10, 10)))
                mask_large[row_id * 256:row_id * 256 + 512, col_id * 256:col_id * 256 + 512] = mask_dilate
                segment = []  # 该实例道路段中存在的点
                for point in point_large:
                    # try:
                    # if point[1] == 512:
                    #     point[1] = 511
                    # if point[0] == 512:
                    #     point[0] = 511
                    if mask_large[point[0], point[1]] == 1:  # graph 中是横轴，纵轴
                        segment.append([int(x) for x in point])
                # if len(segment)>=2:
                ## 由细化掩膜得到线段
                ske = skeletonize(np.ascontiguousarray(mask_large)).astype(np.uint16)
                G = sknw.build_sknw(ske, multi=True)
                seg_point = G2point(G)  ## [array[line1]，array[[point1],[point2]]]
                seg_line_points = []  ##细化得到的所有关键点
                for seg_line in seg_point:
                    seg_line_points.extend(seg_line)

                for point_index, seg_point_i in enumerate(seg_line_points):  # 找到已有点图里是否有重复点
                    duplicate_condition = (
                            np.sqrt(np.sum((np.array(seg_point_i) - np.array(point_large)) ** 2, axis=1)) > 1)
                    if np.all(duplicate_condition):
                        # pass
                #         # 没有重复点，直接添加点到点集中
                        point_large = point_large + [seg_point_i]
                #         # 形成该掩膜内覆盖到的所有点
                        segment = segment + [seg_point_i]
                    else:
                        # 有重复点，用点集中点更新道路段得到地点
                        duplicate_point_index = np.where(duplicate_condition == 0)[0][0]
                        for i, seg_point1 in enumerate(seg_point):
                            for j, seg_point2 in enumerate(seg_point1):
                                if seg_point2 == seg_point_i:
                                    seg_point[i][j] = point_large[duplicate_point_index]
                                    break
                        seg_line_points[point_index] = point_large[duplicate_point_index]

                segment = segment + seg_line_points # 组合实例覆盖点和细化骨架点
                segment = list(map(list, set(map(tuple, segment))))  # 去除segment和seg_point_line中可能存在的重复点
                # 判断该道路实例内是否存在边
                # if len(set([str(item) for item in segment]).intersection(
                #         set([str(item) for item in point_connected]))) >= 2:
                    # pass
                    #已存在边，旅行商连通性
                if len(segment)>2:
                    # 最短距离算法--旅行商
                    segment.sort(key=lambda x: x[0])
                    ## 使用elkai库计算tsp
                    # point_travel = elkai.Coordinates2D({str(i): tuple(point) for i, point in enumerate(segment)})
                    # tsp_path = point_travel.solve_tsp()
                    # shortest_segment = [segment[int(i)] for i in tsp_path[:-1]]
                    # nx-tsp
                    G = nx.complete_graph(len(segment))
                    for i in range(len(segment)):
                        for j in range(i + 1, len(segment)):
                            distance = euclidean(segment[i], segment[j])
                            G[i][j]['weight'] = distance
                    tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
                    shortest_segment = [segment[i] for i in tsp_path]
                else:
                    shortest_segment = segment
                ## 添加道路段得到的连通性点对到图中
                for pp_index, point_pair in enumerate(shortest_segment[1:]):
                    if ([shortest_segment[pp_index], point_pair] not in segments) and (
                            [point_pair, shortest_segment[pp_index]] not in segments) and (
                            point_pair != shortest_segment[pp_index]):
                        segments.append([shortest_segment[pp_index], point_pair])
                point_connected += shortest_segment
                # else:
                # # 未存在边，同时添加实例道路段连通性和旅行商连通性
                # # 组合两个连通性点对列表，去除重复的
                #     if len(segment)>2:
                #             # 最短距离算法--旅行商
                #         segment.sort(key=lambda x: x[0])
                #         ## 使用elkai库计算tsp
                #         # point_travel = elkai.Coordinates2D({str(i): tuple(point) for i, point in enumerate(segment)})
                #         # tsp_path = point_travel.solve_tsp()
                #         # shortest_segment_tsp = [segment[int(i)] for i in tsp_path[:-1]]
                #         ## 使用nx计算tsp
                #         G = nx.complete_graph(len(segment))
                #         for i in range(len(segment)):
                #             for j in range(i + 1, len(segment)):
                #                 distance = euclidean(segment[i], segment[j])
                #                 G[i][j]['weight'] = distance
                #         tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
                #         shortest_segment_tsp = [segment[i] for i in tsp_path]
                #     else:
                #         shortest_segment_tsp = segment
                #     # shortest_segment_ske = seg_point
                #     ## 添加道路段得到的连通性点对到图中
                #     for pp_index, point_pair in enumerate(shortest_segment_tsp[1:]):
                #         if ([shortest_segment_tsp[pp_index], point_pair] not in segments) and (
                #             [point_pair, shortest_segment_tsp[pp_index]] not in segments) and (
                #             point_pair != shortest_segment_tsp[pp_index]):
                #             segments.append([shortest_segment_tsp[pp_index], point_pair])
                #     point_connected += shortest_segment_tsp
                    # for shortest_segment_ske in seg_point:
                    #     for pp_index, point_pair in enumerate(shortest_segment_ske[1:]):
                    #         if ([shortest_segment_ske[pp_index], point_pair] not in segments) and (
                    #                 [point_pair, shortest_segment_ske[pp_index]] not in segments) and (
                    #                 point_pair != shortest_segment_ske[pp_index]):
                    #             segments.append([shortest_segment_ske[pp_index], point_pair])
                    #     point_connected += shortest_segment_ske


                    # point_connected = [list(point) for point in set(tuple(point) for point in point_connected)] # 去除重复点
        wkt = []
        for points in segments:
            # if len(coord_list) > 1:
            line = '(' + ", ".join(
                ' '.join(map(str, (float(point[1]), float(point[0])))) for point in points) + ')'
            wkt.append("LINESTRING {}".format(line))

        all_data = []
        for wkt_one in wkt:
            all_data.append(('AOI_{}_City_{}'.format(str(testid), '0'), wkt_one))
        df = pd.DataFrame(all_data, columns=['ImageId', 'WKT_Pix'])
        df.to_csv(os.path.join(csv_out_dir, 'region_{}'.format(testid)+'.csv'),
                  index=False)


