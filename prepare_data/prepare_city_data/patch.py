# 将2048*2048的png切为512大小的图像
# 同时切对应的gtpng和graph
## 遍历切img，对应的gt坐标
###pickle中记录的是每个点的临点，所以每条边会被记录两遍，我只要获得在patch范围内的点，判断该点每个临点，如果临点超出范围，找到这条边和边框的交点，用交点替换临点

import pickle
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from shapely.geometry import LineString, Polygon, Point, MultiPoint
from numba import jit
# @jit
def graphpatch(graph, r, c):
    """
    获得在patch范围内的点，判断该点每个临点，如果临点超出范围，找到这条边和边框的交点，用交点替换临点
    :param graph: 点字典  {(454, 454)：[(1106, 1656), (1096, 1696)]} r,c
    :param r: patch左上角行值
    :param c: patch左上角列值
    :return: patch内graph值
    """
    patch_graph = {}
    for point in graph:
        if (point[0]>=r)&(point[0]<=(r+512))&(point[1]>=c)&(point[1]<=(c+512)):
            adjacenet_points = []
            for adjacenet_point in graph[point]:
                if (adjacenet_point[0]>=r)&(adjacenet_point[0]<=(r+512))&(adjacenet_point[1]>=c)&(adjacenet_point[1]<=(c+512)):
                    adjacenet_points.append((adjacenet_point[0]-r,adjacenet_point[1]-c))
                else:
                    rectangle_points = [(r,c),(r+512,c),(r+512,c+512),(r,c+512),(r,c)]
                    p1 = Point(point[0],point[1])
                    p2 = Point(adjacenet_point[0],adjacenet_point[1])
                    line = LineString([(p1.x, p1.y), (p2.x, p2.y)])
                    rectangle = LineString(rectangle_points)
                    intersection = line.intersection(rectangle)# 如果点在边界上，会一起返回
                    if isinstance(intersection, Point):
                        if (intersection.x, intersection.y) != point:
                            adjacenet_points.append((int(intersection.x-r), int(intersection.y-c)))
                    elif isinstance(intersection, MultiPoint):
                        for intersect_point in intersection.geoms:
                            if (intersect_point.x, intersect_point.y) != point:
                                adjacenet_points.append((int(intersect_point.x-r), int(intersect_point.y-c)))
                    else:
                        for intersect_point in intersection.coords:
                            if (intersect_point[0], intersect_point[1]) != point:
                                adjacenet_points.append((int(intersect_point[0]-r), int(intersect_point[1]-c)))
            patch_graph[point[0]-r,point[1]-c]=adjacenet_points
    return patch_graph

if __name__ == '__main__':
    sat_savedir = r'city_scale/20cities_patch/sat'
    gt_savedir = r'city_scale/20cities_patch/gt'
    graph_savedir = r'city_scale/20cities_patch/graph_p'
    os.makedirs(sat_savedir,exist_ok=True)
    os.makedirs(gt_savedir, exist_ok=True)
    os.makedirs(graph_savedir, exist_ok=True)

    for satpath in tqdm(glob(r'city_scale/20cities/*_sat.png')):
        gtpath = satpath.replace('sat', 'gt')
        graphpath = satpath.replace('sat.png', 'graph_gt.pickle')
        sat = np.array(Image.open(satpath))
        gt = np.array(Image.open(gtpath))
        graph = pickle.load(open(graphpath,'rb'))
        for r_id, r in enumerate(range(0,2048-512+1,256)):
            for c_id, c in enumerate(range(0, 2048 - 512 + 1, 256)):
                sat_patch = sat[r:r+512,c:c+512,:]
                gt_patch = gt[r:r+512,c:c+512]
                graph_patch = graphpatch(graph,r,c)
                # graph =  {key: graph[key] for key in graph if key not in graph_patch}
                Image.fromarray(sat_patch,mode='RGB').save(os.path.join(sat_savedir, os.path.basename(satpath)[:-8]+'_{}.png'.format(str(r_id)+str(c_id))))
                Image.fromarray(gt_patch, mode='L').save(
                    os.path.join(gt_savedir, os.path.basename(satpath)[:-8]+'_{}.png'.format(str(r_id)+str(c_id))))
                pickle.dump(graph_patch,open(os.path.join(graph_savedir, os.path.basename(satpath)[:-8]+'_{}.p'.format(str(r_id)+str(c_id))),'wb'))
        # print('End')
