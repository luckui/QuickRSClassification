import geopandas as gpd
import numpy as np
from osgeo import gdal
import os


# 配置参数
input_sample = "temp/sample_selected.shp"  # 输入样本点文件
input_raster = "testTif\\region.tif"  # 输入栅格文件
output_sample = "temp/sample_final.shp"  # 输出最终样本点文件


def extract_raster_values_vectorized(points_gdf, raster_path):
    """
    向量化提取样本点的栅格值
    
    参数:
        points_gdf: GeoDataFrame，包含样本点
        raster_path: 栅格文件路径
    返回:
        更新后的GeoDataFrame，包含所有波段值
    """
    print(f"正在打开栅格: {raster_path}")
    
    # 打开栅格
    raster_ds = gdal.Open(raster_path)
    if raster_ds is None:
        raise ValueError(f"无法打开栅格文件: {raster_path}")
    
    # 获取栅格信息
    band_count = raster_ds.RasterCount
    geotransform = raster_ds.GetGeoTransform()
    
    print(f"栅格波段数: {band_count}")
    print(f"样本点数量: {len(points_gdf)}")
    
    # 地理变换参数
    x_origin = geotransform[0]
    y_origin = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    
    # 向量化读取所有波段
    print("\n正在读取所有波段...")
    y_size = raster_ds.RasterYSize
    x_size = raster_ds.RasterXSize
    all_bands = np.zeros((band_count, y_size, x_size), dtype=np.float32)
    
    for i in range(band_count):
        band = raster_ds.GetRasterBand(i + 1)
        all_bands[i] = band.ReadAsArray()
        print(f"  波段 {i+1}/{band_count} 读取完成")
    
    # 向量化提取点坐标
    print("\n正在提取样本点坐标...")
    geometries = points_gdf.geometry
    
    # 批量提取X和Y坐标（向量化）
    x_coords = np.array([geom.x for geom in geometries])
    y_coords = np.array([geom.y for geom in geometries])
    
    # 向量化计算像素行列号
    print("正在计算像素坐标...")
    col_indices = ((x_coords - x_origin) / pixel_width).astype(int)
    row_indices = ((y_coords - y_origin) / pixel_height).astype(int)
    
    # 检查边界（向量化）
    valid_mask = (
        (col_indices >= 0) & (col_indices < x_size) &
        (row_indices >= 0) & (row_indices < y_size)
    )
    
    n_valid = np.sum(valid_mask)
    n_invalid = len(valid_mask) - n_valid
    
    print(f"\n有效样本点: {n_valid}")
    if n_invalid > 0:
        print(f"警告：{n_invalid} 个样本点超出栅格范围")
    
    # 向量化提取像素值（关键优化！）
    print("\n正在向量化提取像素值...")
    
    # 为所有点创建结果数组
    pixel_values = np.full((band_count, len(points_gdf)), np.nan, dtype=np.float32)
    
    # 只对有效点提取值（向量化索引）
    if n_valid > 0:
        valid_rows = row_indices[valid_mask]
        valid_cols = col_indices[valid_mask]
        
        # 一次性提取所有有效点的所有波段值
        pixel_values[:, valid_mask] = all_bands[:, valid_rows, valid_cols]
    
    print("像素值提取完成！")
    
    # 创建新的GeoDataFrame副本
    result_gdf = points_gdf.copy()
    
    # 向量化添加波段值到属性表
    print("\n正在更新属性表...")
    for i in range(band_count):
        band_name = f'BAND_{i+1}'
        result_gdf[band_name] = pixel_values[i]
        print(f"  {band_name} 添加完成")
    
    # 关闭栅格
    raster_ds = None
    
    # 统计提取结果
    print("\n提取结果统计:")
    for i in range(band_count):
        band_name = f'BAND_{i+1}'
        valid_count = result_gdf[band_name].notna().sum()
        print(f"  {band_name}: {valid_count}/{len(result_gdf)} 个有效值")
    
    return result_gdf


def main():
    """主函数"""
    try:
        print("\n" + "="*60)
        print("向量化样本点栅格值提取")
        print("="*60 + "\n")
        
        # 1. 读取样本点
        print(f"正在读取样本点: {input_sample}")
        points_gdf = gpd.read_file(input_sample)
        
        print(f"样本点CRS: {points_gdf.crs}")
        print(f"样本点数量: {len(points_gdf)}")
        print(f"现有字段: {list(points_gdf.columns)}")
        
        # 2. 向量化提取栅格值
        result_gdf = extract_raster_values_vectorized(points_gdf, input_raster)
        
        # 3. 保存结果
        print(f"\n正在保存结果: {output_sample}")
        
        # 创建输出目录
        output_dir = os.path.dirname(output_sample)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        result_gdf.to_file(output_sample)
        
        print("保存成功！")
        
        print("\n" + "="*60)
        print("采样完成！")
        print("="*60)
        print(f"输入样本点: {input_sample}")
        print(f"输入栅格: {input_raster}")
        print(f"输出样本点: {output_sample}")
        print(f"新增字段: {[col for col in result_gdf.columns if col.startswith('BAND_')]}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

