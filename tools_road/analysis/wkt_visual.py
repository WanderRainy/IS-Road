from matplotlib import image
from matplotlib import pyplot as plt
import pandas
import cv2
import numpy as np
from tqdm import tqdm
import os
from glob import glob
# 可视化并保存文件夹内csv文件
def stretch_contrast(image):
    # Convert the image to grayscale
    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate minimum and maximum intensity values
    min_intensity = np.min(image,axis=(0,1))
    max_intensity = np.max(image,axis=(0,1))

    # Apply stretching
    stretched_image = ((image - min_intensity) / (max_intensity - min_intensity)) * 255

    # Convert the stretched image back to 3 channels
    # stretched_image_colored = cv2.cvtColor(stretched_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    return stretched_image.astype(np.uint8)
# if __name__ == '__main__':
#     csv_dir = r'C:\Users\Rain\Desktop\centerline\RoadSegment\results\exp8_1'
#     save_dir = r'C:\Users\Rain\Desktop\centerline\RoadSegment\results\exp8_1_plt'
#     os.makedirs(save_dir,exist_ok=True)
#     for csv_path in tqdm(glob(os.path.join(csv_dir,'*.csv'))):
#         pred_wkt_path = csv_path
#         img_path = r'C:\Users\Rain\Desktop\centerline\RoadSegment\data\RGB_1.0_meter\{}__rgb.png'.format(os.path.basename(csv_path)[:-4])
#         # img_path = r'C:\Users\Rain\Desktop\centerline\RoadSegment\city_scale\20cities\{}_sat.png'.format(os.path.basename(csv_path)[:-4])
#         wkt = pandas.read_csv(pred_wkt_path)
#         Linestring = wkt.loc[:, "WKT_Pix"].tolist()
#         # to read the image stored in the working directory
#         data = image.imread(img_path)
#         plt.figure(dpi=512, figsize=(4, 4))  # 256*32=8192
#         plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.1, wspace=0.1)
#         for linestring in Linestring:
#             x = []
#             y = []
#             xys = linestring[12:-1].split(',')
#             for xy in xys:
#                 x.append(int(float(xy.split(' ')[-2])))
#                 y.append(int(float(xy.split(' ')[-1])))
#             plt.plot(x, y, color='orange', linewidth=0.25, zorder=1)
#             plt.scatter(x, y, c='yellow', marker='o', s=0.5, linewidths=0, zorder=2)
#         plt.axis('off')
#         plt.imshow(data)
#         plt.savefig(
#             os.path.join(save_dir,os.path.basename(csv_path).replace('.csv', '.png')))  # ,bbox_inches='tight',pad_inches=0
        # plt.show()


# 可视化单个csv文件
# if __name__ == '__main__':
#     # pred_wkt_path = r'C:\Users\Rain\Desktop\Overpass\changjiang12_overpass\4.csv'
#     # pred_wkt_path = r'C:\Users\Rain\Desktop\centerline\RoadSegment\city_scale\20cities_patch\graph_wkt\region_108_00.csv'
#     pred_wkt_path = r'C:\Users\Rain\Desktop\centerline\RoadSegment\results_city\exp7_1_csv\region_49_51.csv'
#     pred_wkt_path = r"C:\Users\Rain\Desktop\centerline\RNGDetPlusPlus\spacenet\Author_results\RNGDet_sn_1m_wkt\AOI_2_Vegas_22.csv"
#     # pred_wkt_path = r'C:\Users\Rain\Desktop\centerline\RoadSegment\city_scale\20cities_test_wkt\region_8.csv'
#     # pred_wkt_path = r'C:\Users\Rain\Desktop\centerline\data_baseline\RGB_1.0_meter_wkt\AOI_2_Vegas_22.csv'
#     pred_wkt_path = r"C:\Users\Rain\Desktop\centerline\RoadSegment\results\exp8_4\AOI_2_Vegas_22.csv"#exp5_2_2_v1
#     # img_path = r'C:\Users\Rain\Desktop\centerline\RoadSegment\city_scale\20cities_patch\sat\region_49_51.png'
#     img_path = r"C:\Users\Rain\Desktop\centerline\Sat2Graph\data\spacenet\RGB_1.0_meter_full\RGB_1.0_meter\AOI_2_Vegas_22__rgb.png"
#
#     # pred_wkt_path = r''
#     wkt = pandas.read_csv(pred_wkt_path)
#     Linestring = wkt.loc[:, "WKT_Pix"].tolist()
#     # to read the image stored in the working directory
#     data = stretch_contrast(image.imread(img_path))
#     plt.figure(dpi=256, figsize=(4, 4))  # 256*32=8192
#     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.1, wspace=0.1)
#     count = 1
#     for linestring in Linestring:
#         x = []
#         y = []
#         xys = linestring[12:-1].split(',')
#         for xy in xys:
#             x.append(int(float(xy.split(' ')[-2])))
#             y.append(int(float(xy.split(' ')[-1])))
#         # plt.plot(x, y, color='orange', linewidth=1, zorder=1)
#         # plt.scatter(x, y, c='yellow', marker='o', s=1, zorder=2)
#         # if count==42:
#         #     plt.plot(x, y, color='red', linewidth=1, zorder=1)
#         #     plt.scatter(x, y, c='yellow', marker='o', s=1, zorder=2)
#         #     plt.axis('off')
#         #     plt.imshow(data)
#         #     # plt.savefig(
#         #     #     r'C:\Users\Rain\Desktop\centerline\Sat2Graph\big.png')  # ,bbox_inches='tight',pad_inches=0
#         #     plt.show()
#         # else:
#             plt.plot(x, y, color='orange', linewidth=1, zorder=1)
#             plt.scatter(x, y, c='yellow', marker='o', s=1, zorder=2)
#         count+=1
#     plt.axis('off')
#     plt.imshow(data)
#     # plt.savefig(
#     #     r'C:\Users\Rain\Desktop\centerline\RoadSegment\results_city\exp7_1_large_csv_wkt\region_49_v3.png')  # ,bbox_inches='tight',pad_inches=0
#     plt.show()

# 查看多类方法的结果(SN)
if __name__ == '__main__':
    for name in tqdm(os.listdir(r'C:\Users\Rain\Desktop\centerline\Sat2Graph\data\spacenet\RGB_1.0_meter_full\RGB_1.0_meter')):
        if '_73_' not in name:
            continue
        label_wkt_path = os.path.join(r'C:\Users\Rain\Desktop\centerline\data_baseline\RGB_1.0_meter_wkt_all_topo', name[:-9]+'.csv')
        gamsnet_wkt_path = os.path.join(r'C:\Users\Rain\Desktop\centerline\GAMSNet\SN\GAMSNet_SN_1m_wkt', name[:-9]+'.csv')
        rngdetpp_wkt_path = os.path.join(r'C:\Users\Rain\Desktop\centerline\RNGDetPlusPlus\spacenet\Author_results\RNGDet_sn_1m_wkt', name[:-9]+'.csv')
        exp5_2_2_wkt_path = os.path.join(r'C:\Users\Rain\Desktop\centerline\RoadSegment\results\csv_exp5_2_2', name[:-9]+'.csv')
        # pred_wkt_path = r'C:\Users\Rain\Desktop\centerline\RNGDetPlusPlus\spacenet\Author_results\RNGDet_sn_1m_wkt\AOI_2_Vegas_30.csv'
        img_path = r'C:\Users\Rain\Desktop\centerline\Sat2Graph\data\spacenet\RGB_1.0_meter_full\RGB_1.0_meter\{}__rgb.png'.format(name[:-9])

        # gamsnet
        if os.path.exists(gamsnet_wkt_path): # 部分图片没有预测出csv
            gams_wkt = pandas.read_csv(gamsnet_wkt_path)
        else:
            continue
        Linestring = gams_wkt.loc[:, "WKT_Pix"].tolist()
        # to read the image stored in the working directory
        data = image.imread(img_path)
        plt.figure(dpi=256, figsize=(7, 3.5))  # 256*32=8192
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.1, wspace=0.1)
        for linestring in Linestring:
            x = []
            y = []
            xys = linestring[12:-1].split(',')
            for xy in xys:
                x.append(int(float(xy.split(' ')[-2])))
                y.append(int(float(xy.split(' ')[-1])))
            plt.subplot(1,4,2)
            plt.plot(x, y, color='orange', linewidth=1, zorder=1)
            plt.scatter(x, y, c='yellow', marker='o', s=1, zorder=2)
            plt.axis('off')
            plt.imshow(data)

        # RNGDet
        if os.path.exists(rngdetpp_wkt_path):
            rngdetpp_wkt = pandas.read_csv(rngdetpp_wkt_path)
        else:
            continue
        Linestring = rngdetpp_wkt.loc[:, "WKT_Pix"].tolist()
        # to read the image stored in the working directory
        data = image.imread(img_path)
        # plt.figure(dpi=256, figsize=(32, 32))  # 256*32=8192
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.1, wspace=0.1)
        for linestring in Linestring:
            x = []
            y = []
            xys = linestring[12:-1].split(',')
            for xy in xys:
                x.append(int(float(xy.split(' ')[-2])))
                y.append(int(float(xy.split(' ')[-1])))
            plt.subplot(1, 4, 3)
            plt.plot(x, y, color='orange', linewidth=1, zorder=1)
            plt.scatter(x, y, c='yellow', marker='o', s=1, zorder=2)
            plt.axis('off')
            plt.imshow(data)

        # roadsegment
        if os.path.exists(exp5_2_2_wkt_path):
            exp5_2_2_wkt = pandas.read_csv(exp5_2_2_wkt_path)
        else:
            continue
        Linestring = exp5_2_2_wkt.loc[:, "WKT_Pix"].tolist()
        # to read the image stored in the working directory
        data = image.imread(img_path)
        # plt.figure(dpi=256, figsize=(32, 32))  # 256*32=8192
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.1, wspace=0.1)
        for linestring in Linestring:
            x = []
            y = []
            xys = linestring[12:-1].split(',')
            for xy in xys:
                x.append(int(float(xy.split(' ')[-2])))
                y.append(int(float(xy.split(' ')[-1])))
            plt.subplot(1, 4, 4)
            plt.plot(x, y, color='orange', linewidth=1, zorder=1)
            plt.scatter(x, y, c='yellow', marker='o', s=1, zorder=2)
            plt.axis('off')
            plt.imshow(data)
        # plt.savefig(
        #     r'C:\Users\Rain\Desktop\centerline\Sat2Graph\big.png')  # ,bbox_inches='tight',pad_inches=0
        # plt.show()

        # label
        if os.path.exists(label_wkt_path):  # 部分图片没有预测出csv
            label_wkt = pandas.read_csv(label_wkt_path)
        else:
            continue
        Linestring = label_wkt.loc[:, "WKT_Pix"].tolist()
        # to read the image stored in the working directory
        data = image.imread(img_path)
        # plt.figure(dpi=256, figsize=(32, 32))  # 256*32=8192
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.1, wspace=0.1)
        for linestring in Linestring:
            x = []
            y = []
            xys = linestring[12:-1].split(',')
            for xy in xys:
                x.append(int(float(xy.split(' ')[-2])))
                y.append(400 - int(float(xy.split(' ')[-1])))
            plt.subplot(1, 4, 1)
            plt.plot(x, y, color='orange', linewidth=1, zorder=1)
            plt.scatter(x, y, c='yellow', marker='o', s=1, zorder=2)
            plt.axis('off')
            plt.imshow(data)
        plt.savefig(
            r'C:\Users\Rain\Desktop\centerline\tools\vis_paper_fig_spacenet\{}.png'.format(name[:-9]))  # ,bbox_inches='tight',pad_inches=0
        plt.show()
        plt.clf()
        # plt.show()

# 查看多类方法的结果(City-Scale)
