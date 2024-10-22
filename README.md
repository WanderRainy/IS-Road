# IS-Road
open code for IS-RoadDet

This project is based on mmdetection.
## 数据集准备
模型需要实例级的道路标注，因此需要从开源数据集（SpaceNet和cityscale都提供的是linestring）转化为coco格式的道路实例标注。
```python
data/prepare_data/prepare_data/create_label.py # 用于转化道路网linestring为道路实例linstring
data/prepare_data/prepare_data/graph2instance_mask.py #用于转化道路实例linestring为道路instance掩膜
data/prepare_data/prepare_data/shapes_to_coco.py #道路实例掩膜转化为coco instance格式
```
获取交叉点的掩膜标注
```python
/data1/yry22/Vector/IS-RoadDet/mmdetection/data/prepare_data/prepare_data/pointmask_maker.py
```
准备好的数据集【百度网盘，
链接：https://pan.baidu.com/s/1b9TDG6K5fRKD1kzUfIbVVQ?pwd=g881 
提取码：g881】

## 模型训练
mmdetection框架中的训练，分别检查配置文件中数据集路径即可。
```python
CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/***.py --work-dir ./work_dirs/exp/
```
## 模型推理
```python
python tools/test.py configs/***.py work_dirs/exp/epoch_24.pth --work-dir work_dirs/exp --show --show-dir test_out/
```
推理时注意预测交叉点掩膜输出路径需在 mmdet/models/detectors/cascade_rcnn_point.py 260行修改,其他架构类似

## ICS处理
将得到的道路实例段和交叉点连接得到最终的道路图
```python
ics/point_segment2graph_v5.py
```

## 本项目中一些分析脚本及评估代码
```python
tools_road
```
# Thanks
RNGDet,Sat2Graph
