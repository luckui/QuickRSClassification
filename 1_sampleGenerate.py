from osgeo import gdal, ogr, osr
import geopandas as gpd
import numpy as np


# 对输入栅格进行样本点生成，注意读取nodatavalue，nodata区域不生成样本点
# 结果输出shp文件

input_raster = "testTif\\region.tif"
output_point = "temp/sample.shp"
grid_size = 250 # 采样距离，单位为米


def generate_sample_points(input_raster, output_point, grid_size):
    """
    从栅格数据生成样本点，避开nodata区域（向量化高性能版本）
    
    参数:
        input_raster: 输入栅格文件路径
        output_point: 输出shapefile路径
        grid_size: 采样间隔（米）
    """
    
    # 打开栅格数据
    raster_ds = gdal.Open(input_raster)
    if raster_ds is None:
        print(f"无法打开栅格文件: {input_raster}")
        return
    
    # 获取栅格信息
    band_count = raster_ds.RasterCount
    geotransform = raster_ds.GetGeoTransform()
    projection = raster_ds.GetProjection()
    
    # 获取栅格范围
    x_size = raster_ds.RasterXSize
    y_size = raster_ds.RasterYSize
    
    # 地理转换参数
    x_origin = geotransform[0]
    y_origin = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    
    # 获取第一波段的nodata值
    band1 = raster_ds.GetRasterBand(1)
    nodata_value = band1.GetNoDataValue()
    
    print(f"栅格大小: {x_size} x {y_size}")
    print(f"波段数量: {band_count}")
    print(f"NoData值: {nodata_value}")
    print(f"像素分辨率: {pixel_width} x {pixel_height}")
    
    # 判断坐标系类型
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    is_geographic = srs.IsGeographic()
    
    # 如果是地理坐标系，需要将米转换为度
    if is_geographic:
        center_lat = y_origin + (y_size / 2) * pixel_height
        meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))
        meters_per_degree_lat = 111320
        
        grid_size_x = grid_size / meters_per_degree_lon
        grid_size_y = grid_size / meters_per_degree_lat
        
        print(f"坐标系: 地理坐标系 (中心纬度: {center_lat:.6f}°)")
        print(f"采样间隔(度): X={grid_size_x:.8f}°, Y={grid_size_y:.8f}°")
    else:
        grid_size_x = grid_size
        grid_size_y = grid_size
        print(f"坐标系: 投影坐标系")
        print(f"采样间隔(米): {grid_size}")
    
    # 计算采样点的行列间隔
    col_interval = max(1, int(grid_size_x / abs(pixel_width)))
    row_interval = max(1, int(grid_size_y / abs(pixel_height)))
    
    print(f"行列采样间隔: {row_interval} x {col_interval}")
    
    # 向量化生成采样点的行列索引
    rows = np.arange(0, y_size, row_interval)
    cols = np.arange(0, x_size, col_interval)
    
    # 创建网格索引（使用meshgrid）
    col_grid, row_grid = np.meshgrid(cols, rows)
    
    # 展平为一维数组
    row_indices = row_grid.ravel()
    col_indices = col_grid.ravel()
    
    print(f"初始采样点数: {len(row_indices)}")
    print("正在读取所有波段数据...")
    
    # 向量化读取所有波段数据
    all_bands_data = np.zeros((band_count, y_size, x_size), dtype=np.float32)
    for i in range(band_count):
        band = raster_ds.GetRasterBand(i + 1)
        all_bands_data[i] = band.ReadAsArray()
    
    print("正在向量化提取像素值...")
    
    # 向量化提取所有采样点的所有波段值 - 关键优化！
    # shape: (band_count, n_points)
    pixel_values = all_bands_data[:, row_indices, col_indices]
    
    # 向量化计算地理坐标
    x_coords = x_origin + col_indices * pixel_width + pixel_width / 2
    y_coords = y_origin + row_indices * pixel_height + pixel_height / 2
    
    # 向量化计算网格倍数标记（1x、2x、3x、4x）
    row_steps = row_indices // row_interval
    col_steps = col_indices // col_interval
    
    grid_1x = np.ones(len(row_indices), dtype=int)  # 所有点都在1倍网格上
    grid_2x = ((row_steps % 2 == 0) & (col_steps % 2 == 0)).astype(int)
    grid_3x = ((row_steps % 3 == 0) & (col_steps % 3 == 0)).astype(int)
    grid_4x = ((row_steps % 4 == 0) & (col_steps % 4 == 0)).astype(int)
    
    # 创建掩码：过滤nodata值（任一波段为nodata则排除）
    if nodata_value is not None:
        valid_mask = ~np.any(pixel_values == nodata_value, axis=0)
    else:
        valid_mask = ~np.any(np.isnan(pixel_values), axis=0)
    
    # 过滤NaN值
    valid_mask &= ~np.any(np.isnan(pixel_values), axis=0)
    
    # 应用掩码
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    pixel_values = pixel_values[:, valid_mask]  # shape: (band_count, n_valid_points)
    grid_1x = grid_1x[valid_mask]
    grid_2x = grid_2x[valid_mask]
    grid_3x = grid_3x[valid_mask]
    grid_4x = grid_4x[valid_mask]
    
    valid_points = len(x_coords)
    print(f"有效采样点数: {valid_points}")
    
    # 创建输出目录（如果不存在）
    import os
    output_dir = os.path.dirname(output_point)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建输出shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    
    # 如果文件已存在，删除它
    if os.path.exists(output_point):
        driver.DeleteDataSource(output_point)
    
    # 创建数据源和图层
    out_ds = driver.CreateDataSource(output_point)
    if out_ds is None:
        print(f"无法创建输出文件: {output_point}")
        return
    
    out_layer = out_ds.CreateLayer('sample_points', srs, ogr.wkbPoint)
    if out_layer is None:
        print("无法创建图层")
        return
    
    # 添加属性字段
    out_layer.CreateField(ogr.FieldDefn('ID', ogr.OFTInteger))
    
    field_x = ogr.FieldDefn('X', ogr.OFTReal)
    field_x.SetPrecision(8)
    out_layer.CreateField(field_x)
    
    field_y = ogr.FieldDefn('Y', ogr.OFTReal)
    field_y.SetPrecision(8)
    out_layer.CreateField(field_y)
    
    # 添加分类字段
    field_classname = ogr.FieldDefn('ClassName', ogr.OFTString)
    field_classname.SetWidth(50)
    out_layer.CreateField(field_classname)
    
    out_layer.CreateField(ogr.FieldDefn('ClassValue', ogr.OFTInteger))
    
    # 添加网格倍数标记字段（用于抽稀）
    out_layer.CreateField(ogr.FieldDefn('Grid1x', ogr.OFTInteger))
    out_layer.CreateField(ogr.FieldDefn('Grid2x', ogr.OFTInteger))
    out_layer.CreateField(ogr.FieldDefn('Grid3x', ogr.OFTInteger))
    out_layer.CreateField(ogr.FieldDefn('Grid4x', ogr.OFTInteger))
    
    # 为每个波段创建字段
    for i in range(band_count):
        field = ogr.FieldDefn(f'BAND_{i+1}', ogr.OFTReal)
        field.SetPrecision(6)
        out_layer.CreateField(field)
    
    print("正在写入shapefile...")
    
    # 向量化写入要素（批量创建）
    layer_defn = out_layer.GetLayerDefn()
    
    for i in range(valid_points):
        # 创建点几何
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(float(x_coords[i]), float(y_coords[i]))
        
        # 创建要素
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(point)
        feature.SetField('ID', i)
        feature.SetField('X', float(x_coords[i]))
        feature.SetField('Y', float(y_coords[i]))
        feature.SetField('ClassName', '')
        feature.SetField('ClassValue', 0)
        feature.SetField('Grid1x', int(grid_1x[i]))
        feature.SetField('Grid2x', int(grid_2x[i]))
        feature.SetField('Grid3x', int(grid_3x[i]))
        feature.SetField('Grid4x', int(grid_4x[i]))
        
        # 设置所有波段的值
        for band_idx in range(band_count):
            feature.SetField(f'BAND_{band_idx+1}', float(pixel_values[band_idx, i]))
        
        out_layer.CreateFeature(feature)
        feature = None
    
    print(f"完成！共生成 {valid_points} 个有效样本点，每点包含 {band_count} 个波段值")
    
    # 关闭数据集
    raster_ds = None
    out_ds = None


# 执行样本点生成
if __name__ == "__main__":
    generate_sample_points(input_raster, output_point, grid_size)

