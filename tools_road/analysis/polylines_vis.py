import os
from glob import glob
from tqdm import tqdm
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_polylines(pil_img, polylines):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for polyline,c in zip(polylines, colors):
        polyline = np.stack((np.array(polyline)[:, 0], np.array(polyline)[:, 1]))
        # ax.plot(polyline)
        ax.plot(polyline[0],polyline[1], color=c, linewidth=3)
    plt.axis('off')
    plt.show()
    # plt.savefig('samples/sn_20022_exp1_1result.png')
if __name__ == '__main__':
    graph_dir = r'C:\Users\Rain\Desktop\centerline\RoadSegment\data\graph'
    img_dir = r'C:\Users\Rain\Desktop\centerline\RoadSegment\data\RGB_1.0_meter'
    for graph_path in tqdm(glob(os.path.join(graph_dir,'*.json'))):
        if os.path.basename(graph_path)[:-5].split('_')[-1]=='22':
            graph = json.load(open(graph_path))
            polylines = [polyline['vertices_simplify'] for polyline in graph['edges']]
            img = Image.open(os.path.join(img_dir,os.path.basename(graph_path)[:-5]+'__rgb.png'))
            plot_polylines(img, polylines)


