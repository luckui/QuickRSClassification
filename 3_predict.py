from osgeo import gdal
import numpy as np
import joblib
import os
from datetime import datetime


# 配置参数
model_path = "temp/lgbm_model.pkl"  # 训练好的模型文件
input_raster = "testTif/region.tif"  # 输入栅格
output_raster = "temp/predicted_region_lgbm.tif"  # 输出预测结果
batch_size = 100000  # 批处理大小（像素数量），防止内存溢出


def predict_raster(model_path, input_raster, output_raster, batch_size=100000):
    """
    对栅格数据进行分类预测（向量化高性能版本）
    
    参数:
        model_path: 模型文件路径
        input_raster: 输入栅格路径
        output_raster: 输出栅格路径
        batch_size: 批处理大小
    """
    print("\n" + "="*60)
    print("栅格分类预测")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 加载模型
    print(f"正在加载模型: {model_path}")
    model = joblib.load(model_path)
    print(f"模型加载成功！")
    print(f"模型类型: {type(model).__name__}")
    
    if hasattr(model, 'n_estimators'):
        print(f"树的数量: {model.n_estimators}")
    if hasattr(model, 'n_features_in_'):
        print(f"特征数量: {model.n_features_in_}")
    
    # 2. 打开输入栅格
    print(f"\n正在打开栅格: {input_raster}")
    raster_ds = gdal.Open(input_raster)
    if raster_ds is None:
        raise ValueError(f"无法打开栅格文件: {input_raster}")
    
    # 获取栅格信息
    band_count = raster_ds.RasterCount
    x_size = raster_ds.RasterXSize
    y_size = raster_ds.RasterYSize
    geotransform = raster_ds.GetGeoTransform()
    projection = raster_ds.GetProjection()
    
    # 获取nodata值
    band1 = raster_ds.GetRasterBand(1)
    nodata_value = band1.GetNoDataValue()
    
    print(f"栅格大小: {x_size} x {y_size}")
    print(f"波段数量: {band_count}")
    print(f"像素总数: {x_size * y_size:,}")
    print(f"NoData值: {nodata_value}")
    
    # 验证波段数量与模型特征数量一致
    if hasattr(model, 'n_features_in_') and model.n_features_in_ != band_count:
        raise ValueError(f"模型需要 {model.n_features_in_} 个特征，但栅格只有 {band_count} 个波段")
    
    # 3. 向量化读取所有波段数据
    print(f"\n正在读取所有波段数据...")
    all_bands = np.zeros((band_count, y_size, x_size), dtype=np.float32)
    
    for i in range(band_count):
        band = raster_ds.GetRasterBand(i + 1)
        all_bands[i] = band.ReadAsArray()
        print(f"  波段 {i+1}/{band_count} 读取完成")
    
    print("所有波段读取完成！")
    
    # 4. 向量化重塑数据：(bands, height, width) -> (n_pixels, bands)
    print("\n正在重塑数据...")
    # 转置并重塑：(bands, height, width) -> (height, width, bands) -> (n_pixels, bands)
    all_bands_transposed = np.transpose(all_bands, (1, 2, 0))  # (height, width, bands)
    X_all = all_bands_transposed.reshape(-1, band_count)  # (n_pixels, bands)
    
    print(f"数据形状: {X_all.shape} (像素数 x 波段数)")
    
    # 5. 创建nodata掩码（向量化）
    print("\n正在创建NoData掩码...")
    if nodata_value is not None:
        # 任一波段为nodata则整个像素为nodata
        valid_mask = ~np.any(X_all == nodata_value, axis=1)
    else:
        # 检查NaN值
        valid_mask = ~np.any(np.isnan(X_all), axis=1)
    
    # 同时过滤NaN
    valid_mask &= ~np.any(np.isnan(X_all), axis=1)
    
    n_valid = np.sum(valid_mask)
    n_nodata = len(valid_mask) - n_valid
    
    print(f"有效像素: {n_valid:,} ({n_valid/len(valid_mask)*100:.2f}%)")
    print(f"NoData像素: {n_nodata:,} ({n_nodata/len(valid_mask)*100:.2f}%)")
    
    # 6. 向量化预测（分批处理）
    print(f"\n正在进行预测（批处理大小: {batch_size:,}）...")
    
    # 初始化预测结果数组（使用255作为nodata标记，0是有效类别）
    predictions = np.full(len(valid_mask), 255, dtype=np.uint8)
    
    # 提取有效像素
    X_valid = X_all[valid_mask]
    
    # 分批预测
    n_batches = int(np.ceil(len(X_valid) / batch_size))
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(X_valid))
        
        X_batch = X_valid[start_idx:end_idx]
        pred_batch = model.predict(X_batch)
        
        # 将批次预测结果放回有效像素位置
        valid_indices = np.where(valid_mask)[0][start_idx:end_idx]
        predictions[valid_indices] = pred_batch
        
        print(f"  批次 {batch_idx+1}/{n_batches} 完成 ({end_idx:,}/{len(X_valid):,} 像素)")
    
    print("预测完成！")
    
    # 7. 向量化重塑预测结果：(n_pixels,) -> (height, width)
    print("\n正在重塑预测结果...")
    prediction_map = predictions.reshape(y_size, x_size)
    
    # 统计预测类别分布（不包括nodata）
    valid_predictions = predictions[valid_mask]
    unique_classes, class_counts = np.unique(valid_predictions, return_counts=True)
    print(f"\n预测类别分布:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  类别 {cls}: {count:,} 像素 ({count/n_valid*100:.2f}%)")
    
    # 8. 保存输出栅格
    print(f"\n正在保存预测结果: {output_raster}")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_raster)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建输出栅格
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_raster,
        x_size,
        y_size,
        1,  # 单波段输出
        gdal.GDT_Byte,  # 字节类型（0-255）
        options=['COMPRESS=LZW', 'TILED=YES']  # 压缩选项
    )
    
    if out_ds is None:
        raise ValueError(f"无法创建输出文件: {output_raster}")
    
    # 设置地理变换和投影
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    # 写入数据
    out_band = out_ds.GetRasterBand(1)
    
    # 设置nodata值为255（这样0可以是有效类别）
    out_band.SetNoDataValue(255)
    
    out_band.WriteArray(prediction_map)
    out_band.FlushCache()
    
    # 计算统计信息
    out_band.ComputeStatistics(False)
    
    # 关闭数据集
    out_ds = None
    raster_ds = None
    
    print("预测结果保存成功！")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")


def main():
    """主函数"""
    try:
        predict_raster(model_path, input_raster, output_raster, batch_size)
        
        print("="*60)
        print("预测任务完成！")
        print("="*60)
        print(f"输入栅格: {input_raster}")
        print(f"输出栅格: {output_raster}")
        print(f"模型文件: {model_path}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

