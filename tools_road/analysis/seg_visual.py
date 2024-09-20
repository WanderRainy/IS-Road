from matplotlib import image
from matplotlib import pyplot as plt
import pandas
import cv2
from PIL import Image
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
if __name__ == '__main__':
    tif_dir = r'C:\Users\Rain\Desktop\centerline\GAMSNet\SN\GAMSNet_SN_1m'
    save_dir = r'C:\Users\Rain\Desktop\centerline\GAMSNet\SN\GAMSNet_SN_1m_seg_plt'
    os.makedirs(save_dir,exist_ok=True)
    for tif_path in tqdm(glob(os.path.join(tif_dir,'*.tif'))):
        pred_path = tif_path
        img_path = r'C:\Users\Rain\Desktop\centerline\Sat2Graph\data\spacenet\RGB_1.0_meter_full\RGB_1.0_meter\{}__rgb.png'.format(os.path.basename(tif_path)[:-4])
        # wkt = pandas.read_csv(tif_path)
        img = stretch_contrast(image.imread(img_path))
        # label = image.imread(tif_path,'L')
        label = np.array(Image.open(tif_path))
        img[label>0.5]=[255,0,0]
        plt.figure(dpi=180, figsize=(4, 4))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.1, wspace=0.1)
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(
                os.path.join(save_dir,os.path.basename(tif_path).replace('.tif', '.png')))  # ,bbox_inches='tight',pad_inches=0
        # plt.show()
        # print('show')

        # Linestring = wkt.loc[:, "WKT_Pix"].tolist()
        # # to read the image stored in the working directory
        # data = image.imread(img_path)
        # plt.figure(dpi=180, figsize=(4, 4))  # 256*32=8192
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.1, wspace=0.1)
        # for linestring in Linestring:
        #     x = []
        #     y = []
        #     xys = linestring[12:-1].split(',')
        #     for xy in xys:
        #         x.append(int(float(xy.split(' ')[-2])))
        #         y.append(int(float(xy.split(' ')[-1])))
        #     plt.plot(x, y, color='orange', linewidth=1, zorder=1)
        #     plt.scatter(x, y, c='yellow', marker='o', s=1, zorder=2)
        # plt.axis('off')
        # plt.imshow(data)
        # # plt.savefig(
        # #     os.path.join(save_dir,os.path.basename(csv_path).replace('.csv', '.png')))  # ,bbox_inches='tight',pad_inches=0
        # plt.show()
