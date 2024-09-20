# 输入语义分割得到的点概率图和道路段实例掩膜，输出graph
# 点概率图格式：'.tif',[0,1],float
## 从点概率图中获得关键点列表
## while 每个道路段：
##      在关键点列表中找到道路段掩膜获得覆盖的关键点,不包含两个及以上关键点的道路段舍去
##      每个道路段掩膜中获得转折点

##      组合转折点和关键点成这个道路段的所有点
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
    point_tif_dir = 'results_point/exp8_4'  # 语义分割得到的关键点概率图文件夹
    segm_json_path = r'results/test.segm8_4.json'
    test_json_path = r'data/Instance_road/test/instances_test.json'  # segm中的imgid和图片名关系
    # point_json_path = r'data/graph/AOI_2_Vegas_458.json'
    csv_out_dir = 'results/exp8_4'
    os.makedirs(csv_out_dir, exist_ok=True)

    # 获取图片名对应的imgid
    test_json = json.load(open(test_json_path, 'rb'))
    id_name = {}
    for image_info in test_json['images']:
        id_name[image_info['file_name'][:-4]] = image_info['id']

    # 读取测试集
    testnames = json.load(open(r'data/dataset.json', 'rb'))['test']
    for testname in tqdm(testnames):
        # if testname != 'AOI_2_Vegas_1014': # 调试时查看一个图片
        #     continue
        ## 从点概率图中获得关键点列表
        testimgid = testname.split('_')[1]+testname.split('_')[-1].zfill(4)
        point_tif_path = os.path.join(point_tif_dir, testimgid+'.tif')
        # point_tif = gdal.Open(point_tif_path).ReadAsArray()
        point_tif = np.array(tifffile.imread(point_tif_path))
        neighborhood = scipy.ndimage.generate_binary_structure(2, 2)
        local_max = (scipy.ndimage.maximum_filter(point_tif, footprint=neighborhood) == point_tif)
        point_tif = local_max&(point_tif > (np.max(point_tif)+np.min(point_tif))/2)
        point_coor = np.where(point_tif)
        point_coor = np.vstack((point_coor[0],point_coor[1])).T
        point_coor = [list(x) for x in point_coor]

        ## 可视化点
        from PIL import Image, ImageDraw
        # point_fig = Image.new('L',(400,400))
        # point_drawer = ImageDraw.Draw(point_fig)
        # for point in point_coor:
        #     point_drawer.point((point[1],point[0]),fill=255)
        # point_fig.show()

        # 获取segm中分割信息
        segm_json = json.load(open(segm_json_path, 'rb'))
        segments = []
        point_connected = []
        masks = [] ### 记录所有对应图像的道路段掩膜
        point_fig = Image.new('L', (400, 400))
        point_drawer = ImageDraw.Draw(point_fig)
        for tempindex, road_instance in enumerate(segm_json):
            imgnameid = int(testname.split('_')[1]) * 10000 + int(testname.split('_')[3])
            if (road_instance['image_id'] == id_name[str(imgnameid)])&(road_instance['score']>0.5):
                mask = maskUtils.decode(road_instance['segmentation']) #max=1

                # 可视化实例道路段掩膜
                # Image.fromarray(mask*255).convert('L').save("temp/{}.png".format(str(road_instance['score'])[:9]))  # 可视化道路掩膜

                # continue
                # Image.fromarray(mask * 255).convert('L').show()
                ## 在关键点列表中找到道路段掩膜获得覆盖的关键点,不包含两个及以上关键点的道路段舍去
                ## 找到掩膜范围内的点
                mask_dilate = skimage.morphology.dilation(mask, np.ones((15, 15)))
                import cv2
                # mask_dilate_vis = mask_dilate.copy()*255
                # for point in point_coor:
                #     mask_dilate_vis[point[0],point[1]] = 128
                # cv2.imshow('img', mask_dilate_vis)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                segment = [] # 该实例道路段中存在的点
                for point in point_coor:
                    # try:
                    if point[1] == 400:
                        point[1] = 399
                    if point[0] == 400:
                        point[0] = 399
                    if mask_dilate[point[0], point[1]] == 1:  # graph 中是横轴，纵轴
                        segment.append([int(x) for x in point])

                # if len(segment) >= 0: # 掩膜范围内找到了两个及以上关键点，满足成segment条件
                # masks.append(mask_dilate)
                ## 每个道路段掩膜中获得转折点
                ske = skeletonize(np.ascontiguousarray(mask_dilate)).astype(np.uint16)
                # cv2.imshow('img',ske.astype(np.uint8)*255)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                G = sknw.build_sknw(ske, multi=True)
                seg_point = G2point(G)## [array[line1]，array[[point1],[point2]]]
                # point_coor= point_coor+seg_point ###补充转折点到关键点点集中
                # ## 组合转折点和关键点成这个道路段的所有点,
                # for seg_point_i in seg_point:
                #     # try:
                #     if segment != []:
                #         if np.all(np.sqrt(np.sum((np.array(seg_point_i) - np.array(point_coor))**2, axis=1))>10):
                #             point_coor = point_coor + [seg_point_i]
                #     else:
                #         point_coor = point_coor + [seg_point_i]
                # 添加非重复点到点集中，添加连通性到图中
                # for seg_point_line in seg_point:
                seg_line_points = []
                for seg_line in seg_point:
                    seg_line_points.extend(seg_line)
                for point_index, seg_point_i in enumerate(seg_line_points):
                    # 找到已有点图里是否有重复点
                    if point_coor==[]:
                        point_coor = [seg_point_i]
                    else:
                        duplicate_condition = (np.sqrt(np.sum((np.array(seg_point_i) - np.array(point_coor)) ** 2, axis=1)) > 1)
                        if np.all(duplicate_condition):
                            # 没有重复点，直接添加点到点集中
                            point_coor = point_coor + [seg_point_i]
                            # 形成该掩膜内覆盖到的所有点
                            # segment = segment + [seg_point_i]
                        else:
                            # 有重复点，用点集中点更新道路段得到地点
                            duplicate_point_index=np.where(duplicate_condition==0)[0][0]
                            for i, seg_point1 in enumerate(seg_point):
                                for j, seg_point2 in enumerate(seg_point1):
                                    if seg_point2==seg_point_i:
                                        seg_point[i][j] = point_coor[duplicate_point_index]
                                        break
                            seg_line_points[point_index] = point_coor[duplicate_point_index]
                segment = segment +seg_line_points
                segment = list(map(list, set(map(tuple, segment)))) # 去除segment和seg_point_line中可能存在的重复点

                # 判断该道路实例内是否存在边
                if len(set([str(item) for item in segment]).intersection(set([str(item) for item in point_connected])))>=2:
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
                else:
                    # 未存在边，同时添加实例道路段连通性和旅行商连通性
                    ## 组合两个连通性点对列表，去除重复的
                    if len(segment)>2:
                            # 最短距离算法--旅行商
                        segment.sort(key=lambda x: x[0])
                        ## 使用elkai库计算tsp
                        # point_travel = elkai.Coordinates2D({str(i): tuple(point) for i, point in enumerate(segment)})
                        # tsp_path = point_travel.solve_tsp()
                        # shortest_segment_tsp = [segment[int(i)] for i in tsp_path[:-1]]
                        ## 使用nx计算tsp
                        G = nx.complete_graph(len(segment))
                        for i in range(len(segment)):
                            for j in range(i + 1, len(segment)):
                                distance = euclidean(segment[i], segment[j])
                                G[i][j]['weight'] = distance
                        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
                        shortest_segment_tsp = [segment[i] for i in tsp_path]
                    else:
                        shortest_segment_tsp = segment
                    # shortest_segment_ske = seg_point
                    ## 添加道路段得到的连通性点对到图中
                    for pp_index, point_pair in enumerate(shortest_segment_tsp[1:]):
                        if ([shortest_segment_tsp[pp_index], point_pair] not in segments) and (
                            [point_pair, shortest_segment_tsp[pp_index]] not in segments) and (
                            point_pair != shortest_segment_tsp[pp_index]):
                            segments.append([shortest_segment_tsp[pp_index], point_pair])
                    point_connected += shortest_segment_tsp
                    for shortest_segment_ske in seg_point:
                        for pp_index, point_pair in enumerate(shortest_segment_ske[1:]):
                            if ([shortest_segment_ske[pp_index], point_pair] not in segments) and (
                                    [point_pair, shortest_segment_ske[pp_index]] not in segments) and (
                                    point_pair != shortest_segment_ske[pp_index]):
                                segments.append([shortest_segment_ske[pp_index], point_pair])
                        point_connected += shortest_segment_ske
                point_connected = [list(point) for point in set(tuple(point) for point in point_connected)] # 去除重复点
        wkt = []
        for points in segments:
            # if len(coord_list) > 1:
            line = '(' + ", ".join(
                ' '.join(map(str, (float(point[1]), float(point[0])))) for point in points) + ')'
            wkt.append("LINESTRING {}".format(line))

        all_data = []
        for wkt_one in wkt:
            all_data.append((testname, wkt_one))
        df = pd.DataFrame(all_data, columns=['ImageId', 'WKT_Pix'])
        df.to_csv(os.path.join(csv_out_dir, testname+'.csv'),
                  index=False)

# if __name__ == '__main__':
#     # point_tif_dir = 'results_point/exp5_9'  # 语义分割得到的关键点概率图文件夹
#     # segm_json_path = r'results/test.segm_5_9.json'
#     # test_json_path = r'data/Instance_road/test/instances_test.json'  # segm中的imgid和图片名关系
#     # point_json_path = r'data/graph/AOI_2_Vegas_458.json'
#     #   infer train
#     point_tif_dir = 'data/pointmask'  # 语义分割得到的关键点概率图文件夹
#     segm_json_path = r'results/train_infer.segm.json'
#     test_json_path = r'data/Instance_road/train/instances_train.json'  # segm中的imgid和图片名关系
#     csv_out_dir = 'results/train_infer'
#     os.makedirs(csv_out_dir, exist_ok=True)
#
#     # 获取图片名对应的imgid
#     test_json = json.load(open(test_json_path, 'rb'))
#     id_name = {}
#     for image_info in test_json['images']:
#         id_name[image_info['file_name'][:-4]] = image_info['id']
#
#     # 读取测试集
#     testnames = json.load(open(r'data/dataset.json', 'rb'))['train']
#     for testname in tqdm(testnames):
#         # if testname != 'AOI_2_Vegas_1014': # 调试时查看一个图片
#         #     continue
#         ## 从点概率图中获得关键点列表
#         # testimgid = testname.split('_')[1]+testname.split('_')[-1].zfill(4)
#         point_tif_path = os.path.join(point_tif_dir, testname+'.png')
#         # point_tif = gdal.Open(point_tif_path).ReadAsArray()
#         point_tif = np.array(io.imread(point_tif_path))
#         neighborhood = scipy.ndimage.generate_binary_structure(2, 2)
#         local_max = (scipy.ndimage.maximum_filter(point_tif, footprint=neighborhood) == point_tif)
#         point_tif = local_max&(point_tif > (np.max(point_tif)+np.min(point_tif))/2)
#         point_coor = np.where(point_tif)
#         point_coor = np.vstack((point_coor[0],point_coor[1])).T
#         point_coor = [list(x) for x in point_coor]
#
#         ## 可视化点
#         from PIL import Image, ImageDraw
#         # point_fig = Image.new('L',(400,400))
#         # point_drawer = ImageDraw.Draw(point_fig)
#         # for point in point_coor:
#         #     point_drawer.point((point[1],point[0]),fill=255)
#         # point_fig.show()
#
#         # 获取segm中分割信息
#         segm_json = json.load(open(segm_json_path, 'rb'))
#         segments = []
#         point_connected = []
#         masks = [] ### 记录所有对应图像的道路段掩膜
#         point_fig = Image.new('L', (400, 400))
#         point_drawer = ImageDraw.Draw(point_fig)
#         for tempindex, road_instance in enumerate(segm_json):
#             imgnameid = int(testname.split('_')[1]) * 10000 + int(testname.split('_')[3])
#             if (road_instance['image_id'] == id_name[str(imgnameid)])&(road_instance['score']>0.15):
#                 mask = maskUtils.decode(road_instance['segmentation']) #max=1
#
#                 # 可视化实例道路段掩膜
#                 # Image.fromarray(mask*255).convert('L').save("temp/{}.png".format(str(road_instance['score'])[:9]))  # 可视化道路掩膜
#
#                 # continue
#                 # Image.fromarray(mask * 255).convert('L').show()
#                 ## 在关键点列表中找到道路段掩膜获得覆盖的关键点,不包含两个及以上关键点的道路段舍去
#                 ## 找到掩膜范围内的点
#                 mask_dilate = skimage.morphology.dilation(mask, np.ones((15, 15)))
#                 import cv2
#                 # mask_dilate_vis = mask_dilate.copy()*255
#                 # for point in point_coor:
#                 #     mask_dilate_vis[point[0],point[1]] = 128
#                 # cv2.imshow('img', mask_dilate_vis)
#                 # cv2.waitKey(0)
#                 # cv2.destroyAllWindows()
#
#                 segment = [] # 该实例道路段中存在的点
#                 for point in point_coor:
#                     # try:
#                     if point[1] == 400:
#                         point[1] = 399
#                     if point[0] == 400:
#                         point[0] = 399
#                     if mask_dilate[point[0], point[1]] == 1:  # graph 中是横轴，纵轴
#                         segment.append([int(x) for x in point])
#
#                 # if len(segment) >= 0: # 掩膜范围内找到了两个及以上关键点，满足成segment条件
#                 # masks.append(mask_dilate)
#                 ## 每个道路段掩膜中获得转折点
#                 ske = skeletonize(np.ascontiguousarray(mask_dilate)).astype(np.uint16)
#                 # cv2.imshow('img',ske.astype(np.uint8)*255)
#                 # cv2.waitKey(0)
#                 # cv2.destroyAllWindows()
#                 G = sknw.build_sknw(ske, multi=True)
#                 seg_point = G2point(G)## [array[line1]，array[[point1],[point2]]]
#                 # point_coor= point_coor+seg_point ###补充转折点到关键点点集中
#                 # ## 组合转折点和关键点成这个道路段的所有点,
#                 # for seg_point_i in seg_point:
#                 #     # try:
#                 #     if segment != []:
#                 #         if np.all(np.sqrt(np.sum((np.array(seg_point_i) - np.array(point_coor))**2, axis=1))>10):
#                 #             point_coor = point_coor + [seg_point_i]
#                 #     else:
#                 #         point_coor = point_coor + [seg_point_i]
#                 # 添加非重复点到点集中，添加连通性到图中
#                 # for seg_point_line in seg_point:
#                 seg_line_points = []
#                 for seg_line in seg_point:
#                     seg_line_points.extend(seg_line)
#                 for point_index, seg_point_i in enumerate(seg_line_points):
#                     # 找到已有点图里是否有重复点
#                     if point_coor==[]:
#                         point_coor = [seg_point_i]
#                     else:
#                         duplicate_condition = (np.sqrt(np.sum((np.array(seg_point_i) - np.array(point_coor)) ** 2, axis=1)) > 10)
#                         if np.all(duplicate_condition):
#                             # 没有重复点，直接添加点到点集中
#                             point_coor = point_coor + [seg_point_i]
#                             # 形成该掩膜内覆盖到的所有点
#                             # segment = segment + [seg_point_i]
#                         else:
#                             # 有重复点，用点集中点更新道路段得到地点
#                             duplicate_point_index=np.where(duplicate_condition==0)[0][0]
#                             for i, seg_point1 in enumerate(seg_point):
#                                 for j, seg_point2 in enumerate(seg_point1):
#                                     if seg_point2==seg_point_i:
#                                         seg_point[i][j] = point_coor[duplicate_point_index]
#                                         break
#                             seg_line_points[point_index] = point_coor[duplicate_point_index]
#                 segment = segment +seg_line_points
#                 segment = list(map(list, set(map(tuple, segment)))) # 去除segment和seg_point_line中可能存在的重复点
#
#                 # 判断该道路实例内是否存在边
#                 if len(set([str(item) for item in segment]).intersection(set([str(item) for item in point_connected])))>=2:
#                     #已存在边，旅行商连通性
#                     if len(segment)>2:
#                         # 最短距离算法--旅行商
#                         segment.sort(key=lambda x: x[0])
#                         ## 使用elkai库计算tsp
#                         # point_travel = elkai.Coordinates2D({str(i): tuple(point) for i, point in enumerate(segment)})
#                         # tsp_path = point_travel.solve_tsp()
#                         # shortest_segment = [segment[int(i)] for i in tsp_path[:-1]]
#                         # nx-tsp
#                         G = nx.complete_graph(len(segment))
#                         for i in range(len(segment)):
#                             for j in range(i + 1, len(segment)):
#                                 distance = euclidean(segment[i], segment[j])
#                                 G[i][j]['weight'] = distance
#                         tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
#                         shortest_segment = [segment[i] for i in tsp_path]
#                     else:
#                         shortest_segment = segment
#                     ## 添加道路段得到的连通性点对到图中
#                     for pp_index, point_pair in enumerate(shortest_segment[1:]):
#                         if ([shortest_segment[pp_index], point_pair] not in segments) and (
#                                 [point_pair, shortest_segment[pp_index]] not in segments) and (
#                                 point_pair != shortest_segment[pp_index]):
#                             segments.append([shortest_segment[pp_index], point_pair])
#                     point_connected += shortest_segment
#                 else:
#                     # 未存在边，同时添加实例道路段连通性和旅行商连通性
#                     ## 组合两个连通性点对列表，去除重复的
#                     if len(segment)>2:
#                             # 最短距离算法--旅行商
#                         segment.sort(key=lambda x: x[0])
#                         ## 使用elkai库计算tsp
#                         # point_travel = elkai.Coordinates2D({str(i): tuple(point) for i, point in enumerate(segment)})
#                         # tsp_path = point_travel.solve_tsp()
#                         # shortest_segment_tsp = [segment[int(i)] for i in tsp_path[:-1]]
#                         ## 使用nx计算tsp
#                         G = nx.complete_graph(len(segment))
#                         for i in range(len(segment)):
#                             for j in range(i + 1, len(segment)):
#                                 distance = euclidean(segment[i], segment[j])
#                                 G[i][j]['weight'] = distance
#                         tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
#                         shortest_segment_tsp = [segment[i] for i in tsp_path]
#                     else:
#                         shortest_segment_tsp = segment
#                     # shortest_segment_ske = seg_point
#                     ## 添加道路段得到的连通性点对到图中
#                     for pp_index, point_pair in enumerate(shortest_segment_tsp[1:]):
#                         if ([shortest_segment_tsp[pp_index], point_pair] not in segments) and (
#                             [point_pair, shortest_segment_tsp[pp_index]] not in segments) and (
#                             point_pair != shortest_segment_tsp[pp_index]):
#                             segments.append([shortest_segment_tsp[pp_index], point_pair])
#                     point_connected += shortest_segment_tsp
#                     for shortest_segment_ske in seg_point:
#                         for pp_index, point_pair in enumerate(shortest_segment_ske[1:]):
#                             if ([shortest_segment_ske[pp_index], point_pair] not in segments) and (
#                                     [point_pair, shortest_segment_ske[pp_index]] not in segments) and (
#                                     point_pair != shortest_segment_ske[pp_index]):
#                                 segments.append([shortest_segment_ske[pp_index], point_pair])
#                         point_connected += shortest_segment_ske
#                 point_connected = [list(point) for point in set(tuple(point) for point in point_connected)] # 去除重复点
#         wkt = []
#         for points in segments:
#             # if len(coord_list) > 1:
#             line = '(' + ", ".join(
#                 ' '.join(map(str, (float(point[1]), float(point[0])))) for point in points) + ')'
#             wkt.append("LINESTRING {}".format(line))
#
#         all_data = []
#         for wkt_one in wkt:
#             all_data.append((testname, wkt_one))
#         df = pd.DataFrame(all_data, columns=['ImageId', 'WKT_Pix'])
#         df.to_csv(os.path.join(csv_out_dir, testname+'.csv'),
#                   index=False)


