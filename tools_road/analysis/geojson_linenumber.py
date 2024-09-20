import geopandas as gpd
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
def count_line(filepath):
    geoJ = gpd.read_file(filepath)
    return geoJ.index.size


if __name__ == '__main__':
    lines = [0]*134
    geojson_paths = glob.glob(r'F:\SpaceNet3\SN3_roads_train_AOI_2_Vegas\geojson_roads\*.geojson')
    linemax=0
    for geojson_path in tqdm(geojson_paths):
        line_N = count_line(geojson_path)
        if line_N==133:
            print(geojson_path)
        lines[line_N] += 1
        if line_N>linemax:
            linemax = line_N
    print(linemax)
    plt.bar(range(0,134),lines)
    plt.show()

