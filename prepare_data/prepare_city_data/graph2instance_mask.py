# 从data/graph中生成pycococreator的输入

# 读取json文件，画线，保存
# 同时从原始数据data/RGB_1.0_meter 复制原影像到目标文件夹

import os
import json
import numpy as np
from PIL import ImageDraw, Image
from tqdm import tqdm
from glob import glob
import shutil

if __name__ == '__main__':
    graph_dir = r'city_scale/20cities_patch/graph'
    data_dir = r'city_scale/20cities_patch/sat'
    split_data = r'city_scale/data_split.json'

    save_data_dir = r'city_scale/20cities_patch/Instance_road2'
    os.makedirs(os.path.join(save_data_dir,'train','annotations'), exist_ok=True)
    os.makedirs(os.path.join(save_data_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_data_dir, 'test', 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(save_data_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_data_dir, 'validate', 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(save_data_dir, 'validate', 'images'), exist_ok=True)

    split_dict = json.load(open(split_data, 'r'))
    trainidlist, testidlist, validlist= split_dict['train'], split_dict['test'], split_dict['valid']

    # trainlist = [f'region_{tile_index}' for tile_index in trainlist]
    trainlist = []
    testlist = []
    vallist = []
    for r in range(0,7):
        for c in range(0,7):
            trainlist+=[f'region_{tile_index}_{str(r)+str(c)}' for tile_index in trainidlist]
            testlist+=[f'region_{tile_index}_{str(r) + str(c)}' for tile_index in testidlist]
            vallist+=[f'region_{tile_index}_{str(r) + str(c)}' for tile_index in validlist]
            # split_dict['train']=[f'region_{tile_index}_{str(r)+str(c)}.p' for tile_index in split_dict['train']]

    for json_path in tqdm(glob(os.path.join(graph_dir,'*.json'))):
        if os.path.basename(json_path)[:-5] in trainlist:
            jf = json.load(open(json_path, 'r'))
            # imgid = str(int(os.path.basename(json_path).split('_')[1]) * 10000 + int(
            #     os.path.basename(json_path)[:-5].split('_')[3]))  # AOI_2_Vegas_1.json 保存为20001
            imgid = os.path.basename(json_path).split('_')[-2].zfill(3)+os.path.basename(json_path)[:-5].split('_')[-1] # region_0_66.json 保存为00066
            shutil.copy(os.path.join(data_dir,os.path.basename(json_path)[:-5]+'.png'),
                        os.path.join(save_data_dir,'train', 'images'))
            os.rename(os.path.join(save_data_dir,'train', 'images', os.path.basename(json_path)[:-5]+'.png'),
                      os.path.join(save_data_dir,'train', 'images',imgid+'.png'))
            # 准备画框，每条道路段画到一个图上
            for roadsegment in jf['edges']:
                instance_mask = Image.fromarray(np.zeros((512, 512))).convert('L')
                draw = ImageDraw.Draw(instance_mask)
                # p = instance_mask.load()
                for start_point_index, end_point in enumerate(roadsegment['vertices_simplify'][1:]):
                    draw.line([(roadsegment['vertices_simplify'][start_point_index][0],roadsegment['vertices_simplify'][start_point_index][1]),\
                               (end_point[0], end_point[1])], width = 8, fill=255)

                instance_mask.save(os.path.join(save_data_dir,'train', 'annotations', imgid+'_road_'+str(roadsegment['id'])+ '.png'))

        elif os.path.basename(json_path)[:-5] in vallist:
            jf = json.load(open(json_path, 'r'))
            # imgid = str(int(os.path.basename(json_path)[:-5].split('_')[1]) * 10000 + int(
            #     os.path.basename(json_path)[:-5].split('_')[3]))  # AOI_2_Vegas_1.json 保存为20001
            imgid = os.path.basename(json_path).split('_')[-2].zfill(3) + os.path.basename(json_path)[:-5].split('_')[
                -1]  # region_0_66.json 保存为00066
            shutil.copy(os.path.join(data_dir, os.path.basename(json_path)[:-5] + '.png'),
                        os.path.join(save_data_dir, 'validate', 'images'))
            os.rename(os.path.join(save_data_dir, 'validate', 'images', os.path.basename(json_path)[:-5] + '.png'),
                      os.path.join(save_data_dir, 'validate', 'images', imgid + '.png'))
            # 准备画框，每条道路段画到一个图上
            for roadsegment in jf['edges']:
                instance_mask = Image.fromarray(np.zeros((512, 512))).convert('L')
                draw = ImageDraw.Draw(instance_mask)
                # p = instance_mask.load()
                for start_point_index, end_point in enumerate(roadsegment['vertices_simplify'][1:]):
                    draw.line([(roadsegment['vertices_simplify'][start_point_index][0],
                                roadsegment['vertices_simplify'][start_point_index][1]), \
                               (end_point[0], end_point[1])], width=8, fill=255)

                instance_mask.save(
                    os.path.join(save_data_dir, 'validate', 'annotations', imgid + '_road_' + str(roadsegment['id']) + '.png'))

        elif os.path.basename(json_path)[:-5] in testlist:
            jf = json.load(open(json_path, 'r'))
            # imgid = str(int(os.path.basename(json_path)[:-5].split('_')[1]) * 10000 + int(
            #     os.path.basename(json_path)[:-5].split('_')[3]))  # AOI_2_Vegas_1.json 保存为20001
            imgid = os.path.basename(json_path).split('_')[-2].zfill(3) + os.path.basename(json_path)[:-5].split('_')[
                -1]  # region_0_66.json 保存为00066
            shutil.copy(os.path.join(data_dir, os.path.basename(json_path)[:-5] + '.png'),
                        os.path.join(save_data_dir, 'test', 'images'))
            os.rename(os.path.join(save_data_dir, 'test', 'images', os.path.basename(json_path)[:-5] + '.png'),
                      os.path.join(save_data_dir, 'test', 'images', imgid + '.png'))
            # 准备画框，每条道路段画到一个图上
            for roadsegment in jf['edges']:
                instance_mask = Image.fromarray(np.zeros((512, 512))).convert('L')
                draw = ImageDraw.Draw(instance_mask)
                # p = instance_mask.load()
                for start_point_index, end_point in enumerate(roadsegment['vertices_simplify'][1:]):
                    draw.line([(roadsegment['vertices_simplify'][start_point_index][0],
                                roadsegment['vertices_simplify'][start_point_index][1]), \
                               (end_point[0], end_point[1])], width=8, fill=255)

                instance_mask.save(
                    os.path.join(save_data_dir, 'test', 'annotations', imgid + '_road_' + str(roadsegment['id']) + '.png'))
        else:
            raise Exception('未知image')
