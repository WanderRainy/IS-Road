# C:\Users\Rain\Desktop\centerline\RoadSegment\data\graph
# 统计每个图片上存在的道路实例最大数目
import os
import json
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    graph_dir = r'C:\Users\Rain\Desktop\centerline\RoadSegment\data\graph'
    graph_paths = glob(os.path.join(graph_dir, "*.json"))
    # 统计道路实例数目
    # max_polylines=0
    # for graph_path in tqdm(graph_paths):
    #     graph = json.load(open(graph_path))
    #     count_polylines = len(graph['edges'])
    #     if count_polylines>max_polylines:
    #         max_polylines=count_polylines
    # print(max_polylines)
    # 统计道路段实例数目分布
    max_polylines=[0]*192
    for graph_path in tqdm(graph_paths):
        graph = json.load(open(graph_path))
        count_polylines = len(graph['edges'])
        max_polylines[count_polylines] +=1
    plt.bar(range(len(max_polylines)), max_polylines, fc='g')
    plt.show()

    # 画图
    # max_corner = 0
    # max_corner = [0]*45
    # for graph_path in tqdm(graph_paths):
    #     graph = json.load(open(graph_path))
    #     for edge in graph['edges']:
    #         count_corners = len(edge['vertices_simplify'])
    #         # if count_corners>max_corner:
    #         #     max_corner=count_corners
    # # print(max_corner)
    #         max_corner[count_corners] += 1
    # plt.bar(range(len(max_corner)), max_corner, fc='g')
    # plt.show()

