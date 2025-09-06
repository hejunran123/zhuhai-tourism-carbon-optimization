"""
利用真实的1970-2022年碳排放栅格数据重新建模
修正数据使用错误
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_real_carbon_data():
    """分析真实碳排放数据的可用性"""
    print("🔍 分析真实碳排放数据...")
    
    data_dir = Path("/workspace/文件整理/03_原始数据/珠海数据/中国碳排放1KM栅格数据（1970-2022）")
    
    # 获取所有TIFF文件
    tiff_files = list(data_dir.glob("*.tif"))
    
    # 提取年份信息
    years = []
    for file in tiff_files:
        filename = file.name
        # 从文件名中提取年份
        if "GHG_CO2_" in filename:
            try:
                year_part = filename.split("GHG_CO2_")[1].split("_")[0]
                year = int(year_part)
                if year not in years:
                    years.append(year)
            except:
                continue
    
    years.sort()
    
    print(f"✅ 发现真实碳排放数据:")
    print(f"   时间跨度: {min(years)} - {max(years)}")
    print(f"   总年数: {len(years)}年")
    print(f"   文件数量: {len(tiff_files)}个TIFF文件")
    print(f"   空间分辨率: 1KM栅格")
    
    # 检查其他数据源
    other_data = {
        "旅游数据": "/workspace/文件整理/03_原始数据/珠海数据/我国地级市的国内旅游人数和旅游收入数据（2000-2023）",
        "人口数据": "/workspace/文件整理/03_原始数据/珠海数据/全国省市县人口与人口密度数据（1990-2022）",
        "GDP数据": "/workspace/文件整理/03_原始数据/珠海数据/GDP栅格数据（2014-2020）",
        "能源数据": "/workspace/文件整理/03_原始数据/珠海数据/全国各地级市能源消耗量数据（2000-2022）",
        "夜间灯光": "/workspace/文件整理/03_原始数据/珠海数据/夜间灯光数据（2000-2023）"
    }
    
    print(f"\n📊 其他可用数据源:")
    for name, path in other_data.items():
        if os.path.exists(path):
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: 路径不存在")
    
    return years, tiff_files

def check_tourism_data():
    """检查旅游数据"""
    print(f"\n🏖️ 检查旅游数据...")
    
    excel_dir = Path("/workspace/文件整理/03_原始数据/珠海数据/我国地级市的国内旅游人数和旅游收入数据（2000-2023）/excel格式的数据")
    
    if excel_dir.exists():
        excel_files = list(excel_dir.glob("*.xlsx"))
        print(f"   发现Excel文件: {len(excel_files)}个")
        for file in excel_files[:3]:  # 显示前3个
            print(f"     - {file.name}")
        
        # 尝试读取一个文件看看结构
        if excel_files:
            try:
                df = pd.read_excel(excel_files[0])
                print(f"   数据结构预览:")
                print(f"     形状: {df.shape}")
                print(f"     列名: {list(df.columns)[:5]}...")  # 显示前5列
                return True
            except Exception as e:
                print(f"   读取失败: {e}")
                return False
    else:
        print(f"   ❌ 旅游数据目录不存在")
        return False

def check_energy_data():
    """检查能源数据"""
    print(f"\n⚡ 检查能源消耗数据...")
    
    energy_file = Path("/workspace/文件整理/03_原始数据/珠海数据/全国各地级市能源消耗量数据（2000-2022）/各地级市能源消耗量.xlsx")
    
    if energy_file.exists():
        try:
            df = pd.read_excel(energy_file)
            print(f"   ✅ 能源数据文件存在")
            print(f"     形状: {df.shape}")
            print(f"     列名: {list(df.columns)[:5]}...")
            
            # 查找珠海市数据
            if '地级市' in df.columns or '城市' in df.columns or 'city' in df.columns:
                city_col = None
                for col in df.columns:
                    if '市' in col or 'city' in col.lower():
                        city_col = col
                        break
                
                if city_col:
                    zhuhai_data = df[df[city_col].str.contains('珠海', na=False)]
                    print(f"     珠海市数据: {len(zhuhai_data)}条记录")
                    return True
            
            return True
        except Exception as e:
            print(f"   读取失败: {e}")
            return False
    else:
        print(f"   ❌ 能源数据文件不存在")
        return False

def create_data_integration_plan():
    """创建数据整合计划"""
    print(f"\n📋 创建数据整合计划...")
    
    plan = {
        "阶段1": {
            "任务": "真实碳排放数据处理",
            "数据源": "1970-2022年1KM栅格数据",
            "处理方法": [
                "读取TIFF文件",
                "裁剪到珠海市范围", 
                "重采样到合适分辨率",
                "提取年度总排放量"
            ],
            "预期结果": "53年真实碳排放时间序列"
        },
        
        "阶段2": {
            "任务": "多源数据融合",
            "数据源": "旅游、人口、GDP、能源、夜间灯光",
            "处理方法": [
                "统一时间范围",
                "空间配准",
                "数据插值和外推",
                "质量控制"
            ],
            "预期结果": "完整的多变量时间序列"
        },
        
        "阶段3": {
            "任务": "重新建立STIRPAT模型",
            "数据基础": "53年真实数据",
            "建模方法": [
                "贝叶斯时间序列模型",
                "状态空间模型",
                "动态线性模型",
                "鲁棒性验证"
            ],
            "预期结果": "稳定可靠的STIRPAT模型"
        }
    }
    
    for stage, details in plan.items():
        print(f"\n   {stage}: {details['任务']}")
        print(f"     数据源: {details.get('数据源', 'N/A')}")
        print(f"     方法: {', '.join(details.get('处理方法', []))}")
        print(f"     预期: {details.get('预期结果', 'N/A')}")
    
    return plan

def estimate_improvement_potential():
    """评估改进潜力"""
    print(f"\n📈 评估使用真实数据的改进潜力...")
    
    improvements = {
        "样本量": {
            "当前": "23年模拟数据",
            "改进后": "53年真实数据", 
            "提升": "样本量增加130%"
        },
        "数据质量": {
            "当前": "完全模拟，缺乏现实基础",
            "改进后": "真实观测数据，高度可信",
            "提升": "数据可信度质的飞跃"
        },
        "空间分辨率": {
            "当前": "5KM网格，100个点",
            "改进后": "1KM栅格，数千个像元",
            "提升": "空间分辨率提升25倍"
        },
        "模型稳定性": {
            "当前": "交叉验证R² = -0.88（失败）",
            "改进后": "预期R² > 0.5（成功）",
            "提升": "从完全失败到基本成功"
        },
        "政策价值": {
            "当前": "学术练习，无实用价值",
            "改进后": "可用于实际政策制定",
            "提升": "从概念验证到实用工具"
        }
    }
    
    for aspect, details in improvements.items():
        print(f"\n   {aspect}:")
        print(f"     当前状态: {details['当前']}")
        print(f"     改进后: {details['改进后']}")
        print(f"     提升效果: {details['提升']}")
    
    return improvements

def main():
    """主函数"""
    print("🚨 重大发现：我们忽略了真实的碳排放数据！")
    print("=" * 80)
    
    # 1. 分析真实碳排放数据
    years, tiff_files = analyze_real_carbon_data()
    
    # 2. 检查其他数据源
    tourism_available = check_tourism_data()
    energy_available = check_energy_data()
    
    # 3. 创建数据整合计划
    integration_plan = create_data_integration_plan()
    
    # 4. 评估改进潜力
    improvements = estimate_improvement_potential()
    
    # 5. 总结和建议
    print(f"\n🎯 关键发现和建议:")
    print("=" * 50)
    print("❌ 严重错误:")
    print("   我们有53年的真实碳排放数据，却使用了23年的模拟数据")
    print("   这是导致建模失败的根本原因之一")
    
    print(f"\n✅ 立即行动建议:")
    print("1. 停止使用模拟数据，改用真实的1970-2022年碳排放栅格数据")
    print("2. 整合多源真实数据（旅游、人口、GDP、能源等）")
    print("3. 重新建立基于53年真实数据的STIRPAT模型")
    print("4. 利用1KM高分辨率栅格数据改进空间分析")
    
    print(f"\n📊 预期改进效果:")
    print("• 样本量从23年增加到53年（+130%）")
    print("• 数据质量从模拟提升到真实观测")
    print("• 空间分辨率从5KM提升到1KM（25倍）")
    print("• 模型稳定性从失败（R²=-0.88）到成功（预期R²>0.5）")
    print("• 研究价值从学术练习提升到实用工具")
    
    print(f"\n🚀 下一步行动:")
    print("1. 立即开发真实数据处理脚本")
    print("2. 重新设计基于真实数据的建模流程") 
    print("3. 更新整个优化方案")
    print("4. 重新评估项目价值和贡献")

if __name__ == "__main__":
    main()