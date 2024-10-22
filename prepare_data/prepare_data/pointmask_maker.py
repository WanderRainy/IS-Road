# 利用已有的关键点数据，生成关键点掩膜
## 读取关键点数据
## 根据点坐标生成掩膜
## 高斯核生成
## 保存掩膜
import json
import os.path

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter
if __name__ == '__main__':
    mask_save_dir = r'data\pointmask'

    data_dict = json.load(open(r'data/dataset.json', 'rb'))
    imgnames = data_dict['train']+data_dict['validation']+data_dict['test']
    point_num = []
    for imgname in tqdm(imgnames):
        point_json_path = r'data/graph/{}.json'.format(imgname)
        point_json = json.load(open(point_json_path, 'rb'))
        points = []
        for edge in point_json['edges']:
            for point in edge['vertices_simplify']:
                points.append(point)
        ## 去除重复点
        unique_point_set = set(tuple(x) for x in points)
        unique_points = [list(x) for x in unique_point_set]
        # ？一般一个图中有多少个点
        point_num.append(len(unique_points))

        ## 根据点坐标生成掩膜
        point_fig = Image.new('L', (400, 400))
        point_drawer = ImageDraw.Draw(point_fig)
        for point in unique_points:
            point_drawer.point((point[0],point[1]),fill=255)
        ## 高斯核滤波
        # gaussian_point=gaussian_filter(np.array(point_fig),sigma=1,radius=5)
        gaussian_point = point_fig.filter(ImageFilter.Kernel((3,3), kernel=(1,)*9))
        # gaussian_point.show()
        gaussian_point = np.array(gaussian_point)
        gaussian_point[gaussian_point>0] = 255
        Image.fromarray(gaussian_point,'L').save(os.path.join(mask_save_dir,imgname+'.png'))
        # gaussian_point = Image.fromarray(gaussian_point,'F').save(r'C:\Users\Rain\Desktop\temp.tif','TIFF')
        # print('End')
    print('End')
