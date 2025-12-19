import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    make_scorer
)
import joblib
import os
from datetime import datetime
import json


# 配置参数
input_sample = "temp/sample_selected.shp"  # 输入样本点文件
output_model = "temp/rf_gridsearch_model.pkl"  # 输出最优模型文件
output_report = "temp/gridsearch_report.txt"  # 输出网格搜索报告
output_best_params = "temp/best_params.json"  # 输出最优参数JSON文件

# 网格搜索参数范围
param_grid = {
    'n_estimators': [50, 100, 200],  # 树的数量
    'max_depth': [10, 20, 30, None],  # 最大深度
    'min_samples_split': [2, 5, 10],  # 分裂所需最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶节点最小样本数
    'max_features': ['sqrt', 'log2', None],  # 最大特征数
}

# 其他参数
random_state = 42  # 随机种子
test_size = 0.3  # 测试集比例（70%训练，30%验证）
cv_folds = 5  # 交叉验证折数
scoring = 'accuracy'  # 评分指标


def load_sample_data(sample_file):
    """
    加载样本数据
    
    参数:
        sample_file: shapefile路径
    返回:
        X: 特征矩阵
        y: 标签向量
        feature_names: 特征名称列表
    """
    print(f"正在加载样本数据: {sample_file}")
    
    # 读取shapefile
    gdf = gpd.read_file(sample_file)
    
    # 过滤掉未标注的样本（ClassName不为空）
    # 注意：0也是有效的类别标签！
    labeled_data = gdf[gdf['ClassName'] != ''].copy()
    
    if len(labeled_data) == 0:
        print("警告：没有找到已标注的样本（ClassName为空），将使用所有样本")
        labeled_data = gdf.copy()
    
    print(f"总样本数: {len(gdf)}")
    print(f"已标注样本数: {len(labeled_data)}")
    
    # 提取波段特征（BAND_开头的列）
    band_columns = [col for col in labeled_data.columns if col.startswith('BAND_')]
    
    if len(band_columns) == 0:
        raise ValueError("没有找到波段数据列（BAND_*）")
    
    print(f"特征波段数: {len(band_columns)}")
    print(f"特征列: {band_columns}")
    
    # 提取特征和标签
    X = labeled_data[band_columns].values
    y = labeled_data['ClassValue'].values
    
    # 统计类别分布
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print("\n类别分布:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  类别 {cls}: {count} 个样本")
    
    return X, y, band_columns, labeled_data


def grid_search_rf(X, y, feature_names, param_grid, cv_folds, scoring):
    """
    使用网格搜索寻找随机森林最优超参数
    
    参数:
        X: 特征矩阵
        y: 标签向量
        feature_names: 特征名称列表
        param_grid: 参数网格
        cv_folds: 交叉验证折数
        scoring: 评分指标
    返回:
        best_model: 最优模型
        grid_search: 网格搜索对象
        results: 评估结果字典
    """
    print("\n" + "="*60)
    print("开始网格搜索寻找最优超参数")
    print("="*60)
    
    # 数据集划分（70%训练，30%测试）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # 分层采样，保持类别比例
    )
    
    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 计算参数组合总数
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    
    print(f"\n参数搜索空间:")
    for param_name, param_values in param_grid.items():
        print(f"  {param_name}: {param_values}")
    print(f"\n总参数组合数: {total_combinations}")
    print(f"交叉验证折数: {cv_folds}")
    print(f"总训练次数: {total_combinations * cv_folds}")
    
    # 创建基础模型
    base_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,  # 使用所有CPU核心
        verbose=0
    )
    
    # 创建网格搜索对象
    print(f"\n正在进行网格搜索（评分指标: {scoring}）...")
    print("这可能需要较长时间，请耐心等待...\n")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    # 执行网格搜索
    grid_search.fit(X_train, y_train)
    
    print("\n网格搜索完成！")
    
    # 获取最优模型和参数
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n最优参数:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    print(f"\n最优交叉验证得分: {best_score:.4f}")
    
    # 使用最优模型进行预测
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # 计算训练集精度
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\n训练集精度: {train_accuracy:.4f}")
    
    # 计算测试集精度指标
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    print(f"\n测试集评估结果:")
    print(f"  准确率 (Accuracy):  {test_accuracy:.4f}")
    print(f"  精确率 (Precision): {test_precision:.4f}")
    print(f"  召回率 (Recall):    {test_recall:.4f}")
    print(f"  F1分数 (F1-Score):  {test_f1:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n混淆矩阵:")
    print(cm)
    
    # 分类报告
    print(f"\n详细分类报告:")
    report = classification_report(y_test, y_test_pred, zero_division=0)
    print(report)
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n特征重要性 (Top 10):")
    print(feature_importance.head(10).to_string(index=False))
    
    # 网格搜索结果汇总
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results_sorted = cv_results.sort_values('rank_test_score')
    
    print(f"\n网格搜索Top 5参数组合:")
    top5_cols = ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
    print(cv_results_sorted[top5_cols].head(5).to_string(index=False))
    
    # 汇总结果
    results = {
        'best_params': best_params,
        'best_score': best_score,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'feature_importance': feature_importance,
        'cv_results': cv_results_sorted,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'param_grid': param_grid,
        'cv_folds': cv_folds
    }
    
    return best_model, grid_search, results


def save_model(model, output_path):
    """保存模型"""
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n正在保存最优模型到: {output_path}")
    joblib.dump(model, output_path)
    print("模型保存成功！")


def save_best_params(params, output_path):
    """保存最优参数到JSON文件"""
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n正在保存最优参数到: {output_path}")
    
    # 处理None值（JSON不支持None）
    params_json = {k: (v if v is not None else "null") for k, v in params.items()}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(params_json, f, indent=4, ensure_ascii=False)
    
    print("最优参数保存成功！")


def generate_report(results, output_path):
    """生成网格搜索报告"""
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n正在生成网格搜索报告: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("随机森林网格搜索超参数优化报告\n")
        f.write("="*70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        # 数据集信息
        f.write("-"*70 + "\n")
        f.write("数据集信息\n")
        f.write("-"*70 + "\n")
        f.write(f"训练集样本数: {len(results['X_train'])}\n")
        f.write(f"测试集样本数: {len(results['X_test'])}\n")
        f.write(f"特征数量: {results['X_train'].shape[1]}\n")
        f.write(f"类别数量: {len(np.unique(results['y_train']))}\n")
        f.write("\n")
        
        # 网格搜索配置
        f.write("-"*70 + "\n")
        f.write("网格搜索配置\n")
        f.write("-"*70 + "\n")
        f.write(f"交叉验证折数: {results['cv_folds']}\n")
        f.write(f"评分指标: {scoring}\n")
        f.write("\n参数搜索空间:\n")
        for param_name, param_values in results['param_grid'].items():
            f.write(f"  {param_name}: {param_values}\n")
        f.write("\n")
        
        # 最优参数
        f.write("-"*70 + "\n")
        f.write("最优参数\n")
        f.write("-"*70 + "\n")
        for param_name, param_value in results['best_params'].items():
            f.write(f"{param_name}: {param_value}\n")
        f.write(f"\n最优交叉验证得分: {results['best_score']:.4f}\n")
        f.write("\n")
        
        # 训练/测试集评估
        f.write("-"*70 + "\n")
        f.write("训练集/测试集评估\n")
        f.write("-"*70 + "\n")
        f.write(f"训练集准确率: {results['train_accuracy']:.4f}\n")
        f.write(f"测试集准确率: {results['test_accuracy']:.4f}\n")
        f.write(f"测试集精确率: {results['test_precision']:.4f}\n")
        f.write(f"测试集召回率: {results['test_recall']:.4f}\n")
        f.write(f"测试集F1分数: {results['test_f1']:.4f}\n")
        f.write("\n")
        
        # 混淆矩阵
        f.write("-"*70 + "\n")
        f.write("混淆矩阵\n")
        f.write("-"*70 + "\n")
        f.write(str(results['confusion_matrix']))
        f.write("\n\n")
        
        # 详细分类报告
        f.write("-"*70 + "\n")
        f.write("详细分类报告\n")
        f.write("-"*70 + "\n")
        f.write(results['classification_report'])
        f.write("\n")
        
        # 网格搜索Top 10结果
        f.write("-"*70 + "\n")
        f.write("网格搜索Top 10参数组合\n")
        f.write("-"*70 + "\n")
        top10 = results['cv_results'].head(10)
        for idx, row in top10.iterrows():
            f.write(f"排名 {int(row['rank_test_score'])}: ")
            f.write(f"得分={row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})\n")
            f.write(f"  参数: {row['params']}\n")
        f.write("\n")
        
        # 特征重要性
        f.write("-"*70 + "\n")
        f.write("特征重要性排序\n")
        f.write("-"*70 + "\n")
        for idx, row in results['feature_importance'].iterrows():
            f.write(f"{row['feature']}: {row['importance']:.6f}\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("报告结束\n")
        f.write("="*70 + "\n")
    
    print("网格搜索报告生成成功！")


def main():
    """主函数"""
    try:
        print("\n" + "="*60)
        print("随机森林网格搜索超参数优化")
        print("="*60 + "\n")
        
        # 1. 加载样本数据
        X, y, feature_names, labeled_data = load_sample_data(input_sample)
        
        # 2. 网格搜索寻找最优超参数
        best_model, grid_search, results = grid_search_rf(
            X, y, feature_names, param_grid, cv_folds, scoring
        )
        
        # 3. 保存最优模型
        save_model(best_model, output_model)
        
        # 4. 保存最优参数
        save_best_params(results['best_params'], output_best_params)
        
        # 5. 生成网格搜索报告
        generate_report(results, output_report)
        
        print("\n" + "="*60)
        print("网格搜索完成！")
        print("="*60)
        print(f"最优模型文件: {output_model}")
        print(f"最优参数文件: {output_best_params}")
        print(f"搜索报告: {output_report}")
        print(f"\n最优测试集准确率: {results['test_accuracy']:.4f}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
