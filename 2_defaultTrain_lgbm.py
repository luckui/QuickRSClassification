import geopandas as gpd
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib
import os
from datetime import datetime


# 配置参数
input_sample = "temp/sample_selected.shp"  # 输入样本点文件
output_model = "temp/lgbm_model.pkl"  # 输出模型文件
output_report = "temp/lgbm_accuracy_report.txt"  # 输出精度报告

# LightGBM参数
n_estimators = 100  # 树的数量
max_depth = -1  # 最大深度（-1表示无限制）
learning_rate = 0.1  # 学习率
num_leaves = 31  # 叶子节点数
random_state = 42  # 随机种子
test_size = 0.3  # 测试集比例（70%训练，30%验证）
cv_folds = 5  # 交叉验证折数


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


def train_lightgbm(X, y, feature_names):
    """
    训练LightGBM模型并进行评估
    
    参数:
        X: 特征矩阵
        y: 标签向量
        feature_names: 特征名称列表
    返回:
        model: 训练好的模型
        results: 评估结果字典
    """
    print("\n" + "="*60)
    print("开始训练LightGBM模型")
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
    
    # 创建并训练模型
    print(f"\n模型参数:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  num_leaves: {num_leaves}")
    print(f"  random_state: {random_state}")
    
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        n_jobs=-1,  # 使用所有CPU核心
        verbose=1,
        force_col_wise=True  # 避免警告
    )
    
    print("\n正在训练模型...")
    model.fit(X_train, y_train)
    print("模型训练完成！")
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
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
    
    # 交叉验证
    print(f"\n正在进行{cv_folds}折交叉验证...")
    cv_scores = cross_val_score(
        model, X, y, 
        cv=cv_folds, 
        scoring='accuracy',
        n_jobs=-1
    )
    
    print(f"\n交叉验证结果:")
    for i, score in enumerate(cv_scores):
        print(f"  Fold {i+1}: {score:.4f}")
    print(f"  平均值: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 更详细的交叉验证（多个指标）
    print(f"\n详细交叉验证（多指标）...")
    cv_results = cross_validate(
        model, X, y,
        cv=cv_folds,
        scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
        n_jobs=-1
    )
    
    print(f"  准确率: {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std() * 2:.4f})")
    print(f"  精确率: {cv_results['test_precision_weighted'].mean():.4f} (+/- {cv_results['test_precision_weighted'].std() * 2:.4f})")
    print(f"  召回率: {cv_results['test_recall_weighted'].mean():.4f} (+/- {cv_results['test_recall_weighted'].std() * 2:.4f})")
    print(f"  F1分数: {cv_results['test_f1_weighted'].mean():.4f} (+/- {cv_results['test_f1_weighted'].std() * 2:.4f})")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n特征重要性 (Top 10):")
    print(feature_importance.head(10).to_string(index=False))
    
    # 汇总结果
    results = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_detailed': cv_results,
        'feature_importance': feature_importance,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }
    
    return model, results


def save_model(model, output_path):
    """保存模型"""
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n正在保存模型到: {output_path}")
    joblib.dump(model, output_path)
    print("模型保存成功！")


def generate_report(results, output_path):
    """生成精度报告"""
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n正在生成精度报告: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("LightGBM分类精度评估报告\n")
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
        
        # 模型参数
        f.write("-"*70 + "\n")
        f.write("模型参数\n")
        f.write("-"*70 + "\n")
        f.write(f"算法: LightGBM (Light Gradient Boosting Machine)\n")
        f.write(f"树的数量: {n_estimators}\n")
        f.write(f"最大深度: {max_depth}\n")
        f.write(f"学习率: {learning_rate}\n")
        f.write(f"叶子节点数: {num_leaves}\n")
        f.write(f"随机种子: {random_state}\n")
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
        
        # 交叉验证结果
        f.write("-"*70 + "\n")
        f.write(f"{cv_folds}折交叉验证结果\n")
        f.write("-"*70 + "\n")
        for i, score in enumerate(results['cv_scores']):
            f.write(f"Fold {i+1}: {score:.4f}\n")
        f.write(f"\n平均准确率: {results['cv_mean']:.4f}\n")
        f.write(f"标准差: {results['cv_std']:.4f}\n")
        f.write(f"95%置信区间: +/- {results['cv_std'] * 2:.4f}\n")
        f.write("\n")
        
        # 详细交叉验证
        cv_det = results['cv_detailed']
        f.write("-"*70 + "\n")
        f.write("交叉验证详细指标\n")
        f.write("-"*70 + "\n")
        f.write(f"准确率: {cv_det['test_accuracy'].mean():.4f} (+/- {cv_det['test_accuracy'].std() * 2:.4f})\n")
        f.write(f"精确率: {cv_det['test_precision_weighted'].mean():.4f} (+/- {cv_det['test_precision_weighted'].std() * 2:.4f})\n")
        f.write(f"召回率: {cv_det['test_recall_weighted'].mean():.4f} (+/- {cv_det['test_recall_weighted'].std() * 2:.4f})\n")
        f.write(f"F1分数: {cv_det['test_f1_weighted'].mean():.4f} (+/- {cv_det['test_f1_weighted'].std() * 2:.4f})\n")
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
    
    print("精度报告生成成功！")


def main():
    """主函数"""
    try:
        print("\n" + "="*60)
        print("LightGBM分类模型训练")
        print("="*60 + "\n")
        
        # 1. 加载样本数据
        X, y, feature_names, labeled_data = load_sample_data(input_sample)
        
        # 2. 训练模型并评估
        model, results = train_lightgbm(X, y, feature_names)
        
        # 3. 保存模型
        save_model(model, output_model)
        
        # 4. 生成精度报告
        generate_report(results, output_report)
        
        print("\n" + "="*60)
        print("训练完成！")
        print("="*60)
        print(f"模型文件: {output_model}")
        print(f"精度报告: {output_report}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
