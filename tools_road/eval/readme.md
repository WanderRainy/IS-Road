## Pixel 评价指标
为了统一分割出的阈值，和矢量化方法输出的图结构，
1. 统一将预测结果和标签转为wkt格式
2. 在wkt格式上进行pixel评价
```
eval_pixel.py # 输入为wkt和relax值，输出为指标结果
```

## mask2wkt.py
将分割的预测结果或标签（0~1，浮点值，tif）转为wkt格式
- 后缀不是单纯的影像名和tif的话，需在302和306行修改写入wkt中的影像名

## 评价eval指标
eval_apls.py # 评价两个csv中wkt的apls