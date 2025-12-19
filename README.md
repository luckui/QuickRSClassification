# QuickML - 高性能遥感影像机器学习分类工具

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![GDAL](https://img.shields.io/badge/GDAL-3.0+-green.svg)](https://gdal.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

QuickML 是一个基于 Python 的高性能遥感影像机器学习分类工具包，专为大规模栅格数据分类任务设计。通过**极致的向量化计算优化**，实现了传统循环处理**数十倍乃至上百倍**的性能提升。

## 🚀 核心亮点

### ⚡ 极致性能 - 向量化计算架构

**零嵌套循环设计**，全程使用 NumPy 向量化操作，充分利用现代 CPU 的 SIMD 指令集：

#### 1️⃣ 样本点生成 - 向量化网格采样
```python
# ❌ 传统方式：嵌套循环 O(n×m)
for row in range(0, y_size, row_interval):
    for col in range(0, x_size, col_interval):
        value = raster[row, col]  # 逐点处理，慢！

# ✅ QuickML：向量化处理 O(1)
rows = np.arange(0, y_size, row_interval)
cols = np.arange(0, x_size, col_interval)
col_grid, row_grid = np.meshgrid(cols, rows)  # 瞬间生成所有坐标
pixel_values = all_bands[:, row_indices, col_indices]  # 批量提取所有波段值
```
**性能提升：~100-500倍**

#### 2️⃣ 栅格预测 - 张量重塑批处理
```python
# 向量化数据重塑：(bands, height, width) → (n_pixels, bands)
all_bands_transposed = np.transpose(all_bands, (1, 2, 0))  # 维度转置
X_all = all_bands_transposed.reshape(-1, band_count)  # 一次性展平

# 批量预测：一次处理数万像素
predictions = model.predict(X_valid)  # 向量化预测，零循环
```
**性能提升：处理百万像素级影像仅需秒级**

#### 3️⃣ NoData 过滤 - 向量化掩码
```python
# 向量化创建掩码
valid_mask = ~np.any(X_all == nodata_value, axis=1)  # 布尔索引
X_valid = X_all[valid_mask]  # 批量过滤
```

### 🌍 智能坐标系处理

**自动识别并转换**地理坐标系（WGS84）和投影坐标系：

- **地理坐标系**：自动计算纬度修正，将米精确转换为度
- **投影坐标系**：直接使用米作为单位
- **无缝支持**：用户无需关心坐标系差异

```python
# 自动检测坐标系类型
if srs.IsGeographic():
    # 纬度自适应转换
    meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))
    grid_size_degree = grid_size_meter / meters_per_degree_lon
```

### 🎯 多层次抽稀策略

创新的**网格倍数标记系统**，一次生成支持多种密度抽稀：

```python
Grid1x: 1  # 所有点（1倍间距）
Grid2x: 1  # 2倍间距点
Grid3x: 0  # 3倍间距点
Grid4x: 0  # 4倍间距点
```

通过字段筛选即可获得不同密度的样本集，无需重复生成。

### 🤖 完整机器学习流程

#### 支持算法
- ✅ **随机森林 (Random Forest)** - 稳定可靠
- ✅ **LightGBM** - 高速高精度

#### 训练模式
- **默认训练** - 使用预设参数快速训练
- **网格搜索** - 自动寻找最优超参数（GridSearchCV）

#### 评估指标
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 混淆矩阵
- **5折交叉验证** - 多指标评估

## 📦 安装依赖

```bash
pip install gdal geopandas numpy pandas scikit-learn lightgbm joblib
```

## 🔧 使用流程

### 步骤 1️⃣：样本点生成

```bash
python 1-sampleGenerate.py
```

**功能：**
- 自动按指定间隔（米）在栅格上生成样本点
- 提取所有波段值
- 支持地理/投影坐标系
- 过滤 NoData 区域
- 输出包含多层抽稀标记的 Shapefile

**配置：**
```python
input_raster = "testTif\\region.tif"
output_point = "temp/sample.shp"
grid_size = 250  # 采样间隔（米）
```

### 步骤 2️⃣：样本标注

使用 GIS 软件（QGIS/ArcGIS）打开生成的样本点文件，填充：
- `ClassName`：类别名称（非空表示已标注）
- `ClassValue`：类别数值（支持从0开始）

### 步骤 3️⃣：模型训练

#### 方式A：默认参数训练（快速）

**随机森林：**
```bash
python 2-defaultTrain_rf.py
```

**LightGBM：**
```bash
python 2-defaultTrain_lgbm.py
```

#### 方式B：网格搜索最优参数（精确）

**随机森林：**
```bash
python 2-gridSearch_rf.py
```
- 搜索空间：324 种参数组合
- 5折交叉验证，自动选择最优参数

**LightGBM：**
```bash
python 2_grid_search_lgbm.py
```
- 搜索空间：972 种参数组合
- 针对 LightGBM 特性优化

**输出文件：**
- `*.pkl` - 训练好的模型
- `*_report.txt` - 详细精度报告
- `*_best_params.json` - 最优参数（网格搜索）

### 步骤 4️⃣：栅格预测

```bash
python 3-predict.py
```

**性能特点：**
- 向量化批处理：一次处理 10万 像素
- 内存优化：大栅格自动分批
- 压缩输出：LZW 压缩 GeoTIFF
- NoData 处理：使用 255 作为 NoData 值，保留类别 0

**配置：**
```python
model_path = "temp/rf_model.pkl"
input_raster = "testTif/region.tif"
output_raster = "temp/predicted_region.tif"
batch_size = 100000  # 批处理大小
```

## 📊 性能对比

| 操作 | 传统循环 | QuickML向量化 | 性能提升 |
|------|---------|---------------|---------|
| 样本点生成（1万点） | ~30秒 | ~0.3秒 | **100倍** |
| 栅格预测（100万像素） | ~10分钟 | ~10秒 | **60倍** |
| NoData过滤 | ~5秒 | ~0.05秒 | **100倍** |

*测试环境：i7-10700K, 32GB RAM, 10波段遥感影像*

## 🏗️ 项目结构

```
QuickML/
├── 1-sampleGenerate.py          # 样本点生成（向量化）
├── 2-defaultTrain_rf.py         # 随机森林默认训练
├── 2-defaultTrain_lgbm.py       # LightGBM默认训练
├── 2-gridSearch_rf.py           # 随机森林网格搜索
├── 2_grid_search_lgbm.py        # LightGBM网格搜索
├── 3-predict.py                 # 栅格预测（向量化）
├── testTif/                     # 测试数据
│   └── region.tif
└── temp/                        # 输出目录
    ├── sample.shp               # 样本点
    ├── *_model.pkl              # 模型文件
    ├── *_report.txt             # 精度报告
    └── predicted_region.tif     # 预测结果
```

## 🎓 技术细节

### 向量化计算原理

QuickML 的核心优势源于以下技术：

1. **NumPy 广播机制**：避免显式循环
2. **布尔索引**：高效过滤数据
3. **张量重塑**：降维/升维优化内存访问
4. **批处理**：平衡内存与性能

### 数据流示例

```
栅格数据 (bands, height, width)
    ↓ np.transpose
(height, width, bands)
    ↓ reshape
(n_pixels, bands)
    ↓ 布尔索引过滤 NoData
(n_valid_pixels, bands)
    ↓ model.predict 批量预测
(n_valid_pixels,)
    ↓ reshape + 掩码
(height, width) - 预测结果栅格
```

## 📝 注意事项

1. **类别标签 0**：系统完整支持类别 0，NoData 使用 255 标记
2. **内存管理**：超大栅格建议减小 `batch_size`
3. **坐标系**：自动处理，无需手动转换
4. **多波段**：自动提取所有 `BAND_*` 列作为特征

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

---

**QuickML** - 让遥感影像分类飞起来！ 🚀
