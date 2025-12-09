#!/usr/bin/env python
# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

LABEL_NAMES = {
    1: "商业",
    2: "住宅", 
    3: "公共服务",
    4: "科技与工业区",
    5: "教育文化用地"
}
# ==================== 文件读取相关 ====================

def read_shapefile(shp_path: str, encoding: str = 'utf-8', keep_crs: bool = True) -> gpd.GeoDataFrame:
    """
    读取shapefile文件，保留原始投影
    
    参数:
        shp_path: shp文件路径
        encoding: 编码格式，默认utf-8，中文可尝试'gbk'或'gb2312'
        keep_crs: 是否保留原始坐标系，默认True
    
    返回:
        GeoDataFrame对象
    """
    try:
        gdf = gpd.read_file(shp_path, encoding=encoding)
        if keep_crs and gdf.crs is None:
            print("警告: 未检测到坐标系信息")
        return gdf
    except Exception as e:
        raise Exception(f"读取shapefile失败: {e}")


def read_multiple_shapefiles(shp_paths: List[str], encoding: str = 'utf-8') -> gpd.GeoDataFrame:
    """
    读取多个shapefile并合并
    
    参数:
        shp_paths: shp文件路径列表
        encoding: 编码格式
    
    返回:
        合并后的GeoDataFrame
    """
    gdfs = []
    for path in shp_paths:
        gdf = read_shapefile(path, encoding)
        gdfs.append(gdf)
    
    # 合并所有数据
    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    # 使用第一个文件的坐标系
    if gdfs[0].crs is not None:
        merged.set_crs(gdfs[0].crs, inplace=True)
    
    return merged


# ==================== 文件写入相关 ====================

def write_shapefile(gdf: gpd.GeoDataFrame, output_path: str, encoding: str = 'utf-8', 
                   keep_crs: bool = True) -> bool:
    """
    写入shapefile，保留原始投影
    
    参数:
        gdf: GeoDataFrame对象
        output_path: 输出路径
        encoding: 编码格式
        keep_crs: 是否保留原始坐标系
    
    返回:
        是否成功
    """
    try:
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存时保留坐标系
        if keep_crs:
            gdf.to_file(output_path, encoding=encoding)
        else:
            gdf.to_file(output_path, encoding=encoding, crs=None)
        
        return True
    except Exception as e:
        print(f"写入shapefile失败: {e}")
        return False


def convert_format(gdf: gpd.GeoDataFrame, output_path: str, 
                   output_format: str = 'GeoJSON') -> bool:
    """
    转换矢量格式（支持GeoJSON、KML、GPKG等）
    
    参数:
        gdf: GeoDataFrame对象
        output_path: 输出路径
        output_format: 输出格式（'GeoJSON', 'KML', 'GPKG'等）
    
    返回:
        是否成功
    """
    try:
        driver_map = {
            'GeoJSON': 'GeoJSON',
            'geojson': 'GeoJSON',
            'KML': 'KML',
            'kml': 'KML',
            'GPKG': 'GPKG',
            'gpkg': 'GPKG'
        }
        
        driver = driver_map.get(output_format, output_format)
        gdf.to_file(output_path, driver=driver)
        return True
    except Exception as e:
        print(f"格式转换失败: {e}")
        return False


# ==================== 投影相关 ====================

def get_crs_info(gdf: gpd.GeoDataFrame) -> Dict:
    """
    获取坐标系信息
    
    返回:
        坐标系信息字典
    """
    if gdf.crs is None:
        return {"status": "无坐标系信息"}
    
    return {
        "crs": str(gdf.crs),
        "epsg": gdf.crs.to_epsg(),
        "proj4": gdf.crs.to_proj4() if hasattr(gdf.crs, 'to_proj4') else None,
        "wkt": gdf.crs.to_wkt() if hasattr(gdf.crs, 'to_wkt') else None
    }


def transform_crs(gdf: gpd.GeoDataFrame, target_crs: Union[str, int], 
                 inplace: bool = False) -> gpd.GeoDataFrame:
    """
    转换坐标系
    
    参数:
        gdf: GeoDataFrame对象
        target_crs: 目标坐标系（如'EPSG:4326'或4326）
        inplace: 是否就地修改
    
    返回:
        转换后的GeoDataFrame
    """
    if isinstance(target_crs, int):
        target_crs = f'EPSG:{target_crs}'
    
    if inplace:
        gdf.to_crs(target_crs, inplace=True)
        return gdf
    else:
        return gdf.to_crs(target_crs)


def copy_crs_from_file(source_shp: str, target_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    从另一个shp文件复制投影信息
    
    参数:
        source_shp: 源shp文件路径
        target_gdf: 目标GeoDataFrame
    
    返回:
        设置了投影的GeoDataFrame
    """
    source_gdf = read_shapefile(source_shp)
    if source_gdf.crs is not None:
        target_gdf.set_crs(source_gdf.crs, inplace=True)
    return target_gdf


# ==================== 基本统计 ====================

def get_basic_stats(gdf: gpd.GeoDataFrame) -> Dict:
    """
    获取基本统计信息
    
    返回:
        统计信息字典
    """
    stats = {
        "要素数量": len(gdf),
        "字段数量": len(gdf.columns) - 1,  # 减去geometry列
        "坐标系": str(gdf.crs) if gdf.crs else "无",
        "边界范围": gdf.total_bounds.tolist(),
        "几何类型": gdf.geometry.geom_type.value_counts().to_dict()
    }
    return stats


def get_geometry_stats(gdf: gpd.GeoDataFrame) -> Dict:
    """
    获取几何统计信息
    
    返回:
        几何统计字典
    """
    main_type = gdf.geometry.geom_type.mode()[0]
    stats = {"几何类型": main_type}
    
    if 'Polygon' in main_type or 'MultiPolygon' in main_type:
        areas = gdf.geometry.area
        perimeters = gdf.geometry.length
        stats.update({
            "总面积": float(areas.sum()),
            "平均面积": float(areas.mean()),
            "最大面积": float(areas.max()),
            "最小面积": float(areas.min()),
            "面积标准差": float(areas.std()),
            "总周长": float(perimeters.sum()),
            "平均周长": float(perimeters.mean())
        })
    
    elif 'LineString' in main_type or 'MultiLineString' in main_type:
        lengths = gdf.geometry.length
        stats.update({
            "总长度": float(lengths.sum()),
            "平均长度": float(lengths.mean()),
            "最大长度": float(lengths.max()),
            "最小长度": float(lengths.min()),
            "长度标准差": float(lengths.std())
        })
    
    elif 'Point' in main_type or 'MultiPoint' in main_type:
        bounds = gdf.total_bounds
        x_range = bounds[2] - bounds[0]
        y_range = bounds[3] - bounds[1]
        area = x_range * y_range
        stats.update({
            "X坐标范围": [float(bounds[0]), float(bounds[2])],
            "Y坐标范围": [float(bounds[1]), float(bounds[3])],
            "点密度": float(len(gdf) / area if area > 0 else 0)
        })
    
    return stats


def get_attribute_stats(gdf: gpd.GeoDataFrame) -> Dict:
    """
    获取属性字段统计信息
    
    返回:
        属性统计字典
    """
    attr_cols = [col for col in gdf.columns if col != 'geometry']
    stats = {}
    
    for col in attr_cols:
        col_stats = {
            "数据类型": str(gdf[col].dtype),
            "非空值数": int(gdf[col].count()),
            "空值数": int(gdf[col].isna().sum())
        }
        
        if pd.api.types.is_numeric_dtype(gdf[col]):
            col_stats.update({
                "最小值": float(gdf[col].min()) if not gdf[col].isna().all() else None,
                "最大值": float(gdf[col].max()) if not gdf[col].isna().all() else None,
                "平均值": float(gdf[col].mean()) if not gdf[col].isna().all() else None,
                "中位数": float(gdf[col].median()) if not gdf[col].isna().all() else None
            })
        else:
            unique_count = gdf[col].nunique()
            col_stats["唯一值数量"] = int(unique_count)
            if unique_count <= 10:
                col_stats["唯一值列表"] = gdf[col].unique().tolist()
        
        stats[col] = col_stats
    
    return stats


# ==================== 属性字段操作 ====================

def get_field_names(gdf: gpd.GeoDataFrame, include_geometry: bool = False) -> List[str]:
    """
    获取所有属性字段名称
    
    参数:
        gdf: GeoDataFrame对象
        include_geometry: 是否包含geometry字段
    
    返回:
        字段名称列表
    
    示例:
        fields = get_field_names(gdf)
        print(fields)  # ['name', 'area', 'population']
    """
    if include_geometry:
        return gdf.columns.tolist()
    else:
        return [col for col in gdf.columns if col != 'geometry']


def add_field(gdf: gpd.GeoDataFrame, field_name: str, default_value=None, 
             dtype=None, inplace: bool = False) -> gpd.GeoDataFrame:
    """
    添加新的属性字段
    
    参数:
        gdf: GeoDataFrame对象
        field_name: 新字段名称
        default_value: 默认值（可以是单个值或Series）
        dtype: 数据类型（如'int64', 'float64', 'str'等）
        inplace: 是否就地修改
    
    返回:
        添加字段后的GeoDataFrame
    
    示例:
        gdf = add_field(gdf, 'new_field', 0)  # 添加默认值为0的字段
        gdf = add_field(gdf, 'category', 'A')  # 添加字符串字段
    """
    if not inplace:
        gdf = gdf.copy()
    
    gdf[field_name] = default_value
    
    if dtype is not None:
        gdf[field_name] = gdf[field_name].astype(dtype)
    
    return gdf


def delete_field(gdf: gpd.GeoDataFrame, field_name: str, 
                inplace: bool = False) -> gpd.GeoDataFrame:
    """
    删除单个属性字段
    
    参数:
        gdf: GeoDataFrame对象
        field_name: 要删除的字段名
        inplace: 是否就地修改
    
    返回:
        删除字段后的GeoDataFrame
    
    示例:
        gdf = delete_field(gdf, 'old_field')
    """
    if field_name == 'geometry':
        raise ValueError("不能删除geometry字段")
    
    if not inplace:
        gdf = gdf.copy()
    
    if field_name in gdf.columns:
        gdf.drop(columns=[field_name], inplace=True)
    else:
        print(f"警告: 字段 '{field_name}' 不存在")
    
    return gdf


def delete_fields(gdf: gpd.GeoDataFrame, field_names: List[str], 
                 inplace: bool = False) -> gpd.GeoDataFrame:
    """
    删除多个属性字段
    
    参数:
        gdf: GeoDataFrame对象
        field_names: 要删除的字段名列表
        inplace: 是否就地修改
    
    返回:
        删除字段后的GeoDataFrame
    
    示例:
        gdf = delete_fields(gdf, ['field1', 'field2', 'field3'])
    """
    if 'geometry' in field_names:
        raise ValueError("不能删除geometry字段")
    
    if not inplace:
        gdf = gdf.copy()
    
    existing_fields = [f for f in field_names if f in gdf.columns]
    if existing_fields:
        gdf.drop(columns=existing_fields, inplace=True)
    
    missing_fields = [f for f in field_names if f not in gdf.columns]
    if missing_fields:
        print(f"警告: 以下字段不存在: {missing_fields}")
    
    return gdf


def rename_field(gdf: gpd.GeoDataFrame, old_name: str, new_name: str, 
                inplace: bool = False) -> gpd.GeoDataFrame:
    """
    重命名属性字段
    
    参数:
        gdf: GeoDataFrame对象
        old_name: 原字段名
        new_name: 新字段名
        inplace: 是否就地修改
    
    返回:
        重命名后的GeoDataFrame
    
    示例:
        gdf = rename_field(gdf, 'old_name', 'new_name')
    """
    if not inplace:
        gdf = gdf.copy()
    
    if old_name in gdf.columns:
        gdf.rename(columns={old_name: new_name}, inplace=True)
    else:
        print(f"警告: 字段 '{old_name}' 不存在")
    
    return gdf


def calculate_field(gdf: gpd.GeoDataFrame, field_name: str, expression, 
                   inplace: bool = False) -> gpd.GeoDataFrame:
    """
    根据表达式计算字段值
    
    参数:
        gdf: GeoDataFrame对象
        field_name: 目标字段名（新建或更新）
        expression: 计算表达式（函数或字符串）
        inplace: 是否就地修改
    
    返回:
        计算后的GeoDataFrame
    
    示例:
        # 使用lambda函数
        gdf = calculate_field(gdf, 'area_km2', lambda row: row['面积'] / 1000000)
        
        # 使用字符串表达式
        gdf = calculate_field(gdf, 'density', '人口 / 面积')
    """
    if not inplace:
        gdf = gdf.copy()
    
    if callable(expression):
        gdf[field_name] = gdf.apply(expression, axis=1)
    elif isinstance(expression, str):
        # 使用eval计算表达式
        gdf[field_name] = gdf.eval(expression)
    else:
        gdf[field_name] = expression
    
    return gdf


def get_unique_values(gdf: gpd.GeoDataFrame, field_name: str, 
                     sort: bool = True, dropna: bool = True) -> List:
    """
    获取字段的唯一值
    
    参数:
        gdf: GeoDataFrame对象
        field_name: 字段名
        sort: 是否排序
        dropna: 是否删除空值
    
    返回:
        唯一值列表
    
    示例:
        categories = get_unique_values(gdf, 'category')
    """
    if field_name not in gdf.columns:
        raise ValueError(f"字段 '{field_name}' 不存在")
    
    unique_vals = gdf[field_name].unique()
    
    if dropna:
        unique_vals = unique_vals[pd.notna(unique_vals)]
    
    if sort:
        try:
            unique_vals = sorted(unique_vals)
        except TypeError:
            # 如果不能排序（如混合类型），保持原样
            pass
    
    return unique_vals.tolist()


# ==================== 数据处理 ====================

def filter_by_attribute(gdf: gpd.GeoDataFrame, field: str, 
                       condition: str) -> gpd.GeoDataFrame:
    """
    根据属性条件筛选要素
    
    参数:
        gdf: GeoDataFrame对象
        field: 字段名
        condition: 筛选条件（如 "> 100", "== '城市'", "in ['A', 'B']"）
    
    返回:
        筛选后的GeoDataFrame
    
    示例:
        filter_by_attribute(gdf, 'area', '> 100')
        filter_by_attribute(gdf, 'name', "== '北京'")
    """
    query_str = f"{field} {condition}"
    return gdf.query(query_str)


def filter_by_values(gdf: gpd.GeoDataFrame, field: str, values: Union[list, tuple, set],
                    inverse: bool = False) -> gpd.GeoDataFrame:
    """
    根据字段值列表筛选要素（更灵活的筛选）
    
    参数:
        gdf: GeoDataFrame对象
        field: 字段名
        values: 值列表
        inverse: 是否反向筛选（排除这些值）
    
    返回:
        筛选后的GeoDataFrame
    
    示例:
        # 筛选特定类别
        filtered = filter_by_values(gdf, 'category', ['A', 'B', 'C'])
        
        # 排除特定类别
        filtered = filter_by_values(gdf, 'category', ['X', 'Y'], inverse=True)
    """
    if inverse:
        return gdf[~gdf[field].isin(values)]
    else:
        return gdf[gdf[field].isin(values)]


def filter_by_range(gdf: gpd.GeoDataFrame, field: str, 
                   min_val: float = None, max_val: float = None,
                   inclusive: str = 'both') -> gpd.GeoDataFrame:
    """
    根据数值范围筛选要素
    
    参数:
        gdf: GeoDataFrame对象
        field: 字段名
        min_val: 最小值（None表示不限）
        max_val: 最大值（None表示不限）
        inclusive: 是否包含边界值 ('both', 'left', 'right', 'neither')
    
    返回:
        筛选后的GeoDataFrame
    
    示例:
        # 筛选面积在100到1000之间的要素
        filtered = filter_by_range(gdf, 'area', 100, 1000)
        
        # 筛选大于500的要素
        filtered = filter_by_range(gdf, 'area', min_val=500)
    """
    result = gdf.copy()
    
    if min_val is not None and max_val is not None:
        if inclusive == 'both':
            result = result[(result[field] >= min_val) & (result[field] <= max_val)]
        elif inclusive == 'left':
            result = result[(result[field] >= min_val) & (result[field] < max_val)]
        elif inclusive == 'right':
            result = result[(result[field] > min_val) & (result[field] <= max_val)]
        else:  # 'neither'
            result = result[(result[field] > min_val) & (result[field] < max_val)]
    elif min_val is not None:
        result = result[result[field] >= min_val] if inclusive in ['both', 'left'] else result[result[field] > min_val]
    elif max_val is not None:
        result = result[result[field] <= max_val] if inclusive in ['both', 'right'] else result[result[field] < max_val]
    
    return result


def filter_by_geometry(gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, 
                      predicate: str = 'intersects') -> gpd.GeoDataFrame:
    """
    根据空间关系筛选要素
    
    参数:
        gdf: 要筛选的GeoDataFrame
        boundary: 边界GeoDataFrame
        predicate: 空间关系（'intersects', 'within', 'contains'等）
    
    返回:
        筛选后的GeoDataFrame
    """
    # 确保坐标系一致
    if gdf.crs != boundary.crs and boundary.crs is not None:
        boundary = boundary.to_crs(gdf.crs)
    
    return gpd.sjoin(gdf, boundary, predicate=predicate).drop(columns=['index_right'])


def add_area_length_fields(gdf: gpd.GeoDataFrame, inplace: bool = False) -> gpd.GeoDataFrame:
    """
    添加面积/长度字段
    
    参数:
        gdf: GeoDataFrame对象
        inplace: 是否就地修改
    
    返回:
        添加字段后的GeoDataFrame
    """
    if not inplace:
        gdf = gdf.copy()
    
    geom_type = gdf.geometry.geom_type.mode()[0]
    
    if 'Polygon' in geom_type:
        gdf['面积'] = gdf.geometry.area
        gdf['周长'] = gdf.geometry.length
    elif 'LineString' in geom_type:
        gdf['长度'] = gdf.geometry.length
    
    return gdf


def buffer_geometries(gdf: gpd.GeoDataFrame, distance: float, 
                     inplace: bool = False) -> gpd.GeoDataFrame:
    """
    创建缓冲区
    
    参数:
        gdf: GeoDataFrame对象
        distance: 缓冲距离
        inplace: 是否就地修改
    
    返回:
        缓冲后的GeoDataFrame
    """
    if not inplace:
        gdf = gdf.copy()
    
    gdf.geometry = gdf.geometry.buffer(distance)
    return gdf


def dissolve_by_field(gdf: gpd.GeoDataFrame, field: str, 
                     aggfunc: str = 'first') -> gpd.GeoDataFrame:
    """
    按字段融合要素
    
    参数:
        gdf: GeoDataFrame对象
        field: 用于融合的字段
        aggfunc: 聚合函数（'first', 'sum', 'mean'等）
    
    返回:
        融合后的GeoDataFrame
    """
    return gdf.dissolve(by=field, aggfunc=aggfunc)


def clip_by_boundary(gdf: gpd.GeoDataFrame, 
                    boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    按边界裁剪
    
    参数:
        gdf: 要裁剪的GeoDataFrame
        boundary: 裁剪边界
    
    返回:
        裁剪后的GeoDataFrame
    """
    # 确保坐标系一致
    if gdf.crs != boundary.crs and boundary.crs is not None:
        boundary = boundary.to_crs(gdf.crs)
    
    return gpd.clip(gdf, boundary)


def calculate_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    计算要素中心点
    
    返回:
        中心点GeoDataFrame
    """
    centroids = gdf.copy()
    centroids.geometry = gdf.geometry.centroid
    return centroids


def centralize_coordinates(gdf: gpd.GeoDataFrame, method: str = 'centroid',
                          add_fields: bool = True, inplace: bool = False) -> gpd.GeoDataFrame:
    """
    对矢量坐标进行中心化处理（用于莫兰指数等空间统计分析）
    
    参数:
        gdf: GeoDataFrame对象
        method: 中心化方法
            - 'centroid': 使用几何中心点
            - 'mean': 使用坐标平均值
            - 'median': 使用坐标中位数
        add_fields: 是否添加中心化后的坐标字段（X_centered, Y_centered）
        inplace: 是否就地修改
    
    返回:
        添加了中心化坐标的GeoDataFrame
    
    示例:
        # 中心化坐标用于莫兰指数计算
        gdf = centralize_coordinates(gdf, method='centroid', add_fields=True)
        # 之后可以使用 X_centered 和 Y_centered 字段
    """
    if not inplace:
        gdf = gdf.copy()
    
    # 提取中心点坐标
    centroids = gdf.geometry.centroid
    x_coords = centroids.x
    y_coords = centroids.y
    
    # 计算中心位置
    if method == 'mean':
        x_center = x_coords.mean()
        y_center = y_coords.mean()
    elif method == 'median':
        x_center = x_coords.median()
        y_center = y_coords.median()
    else:  # 'centroid'
        x_center = x_coords.mean()
        y_center = y_coords.mean()
    
    if add_fields:
        # 添加原始坐标
        gdf['X'] = x_coords
        gdf['Y'] = y_coords
        # 添加中心化坐标
        gdf['X_centered'] = x_coords - x_center
        gdf['Y_centered'] = y_coords - y_center
        # 保存中心点信息
        gdf.attrs['center_x'] = x_center
        gdf.attrs['center_y'] = y_center
    
    return gdf


def add_centroid_coordinates(gdf: gpd.GeoDataFrame, prefix: str = '', 
                            inplace: bool = False) -> gpd.GeoDataFrame:
    """
    添加几何中心点的X、Y坐标字段
    
    参数:
        gdf: GeoDataFrame对象
        prefix: 字段名前缀（如'centroid_'会生成'centroid_X'和'centroid_Y'）
        inplace: 是否就地修改
    
    返回:
        添加坐标字段后的GeoDataFrame
    
    示例:
        gdf = add_centroid_coordinates(gdf)  # 添加X和Y字段
        gdf = add_centroid_coordinates(gdf, prefix='center_')  # 添加center_X和center_Y字段
    """
    if not inplace:
        gdf = gdf.copy()
    
    centroids = gdf.geometry.centroid
    gdf[f'{prefix}X'] = centroids.x
    gdf[f'{prefix}Y'] = centroids.y
    
    return gdf


def standardize_values(gdf: gpd.GeoDataFrame, field: str, 
                      method: str = 'zscore', inplace: bool = False) -> gpd.GeoDataFrame:
    """
    标准化字段值（用于空间统计分析）
    
    参数:
        gdf: GeoDataFrame对象
        field: 要标准化的字段名
        method: 标准化方法
            - 'zscore': Z-score标准化 (x - mean) / std
            - 'minmax': 最小-最大标准化 (x - min) / (max - min)
            - 'robust': 鲁棒标准化 (x - median) / IQR
        inplace: 是否就地修改
    
    返回:
        添加标准化字段的GeoDataFrame（字段名为原字段名_std）
    
    示例:
        gdf = standardize_values(gdf, 'population', method='zscore')
        # 会添加 population_std 字段
    """
    if not inplace:
        gdf = gdf.copy()
    
    values = gdf[field]
    
    if method == 'zscore':
        standardized = (values - values.mean()) / values.std()
    elif method == 'minmax':
        standardized = (values - values.min()) / (values.max() - values.min())
    elif method == 'robust':
        median = values.median()
        q75, q25 = values.quantile([0.75, 0.25])
        iqr = q75 - q25
        standardized = (values - median) / iqr if iqr != 0 else values - median
    else:
        raise ValueError(f"未知的标准化方法: {method}")
    
    gdf[f'{field}_std'] = standardized
    
    return gdf


# ==================== 空间统计（莫兰指数） ====================

def compute_morans_i(gdf: gpd.GeoDataFrame,
                    field: str,
                    weights: str = 'queen',
                    distance_threshold: float = None,
                    k: int = 8,
                    row_standardize: bool = True,
                    permutations: int = 999,
                    use_centroid: bool = True) -> Dict:
    """
    计算全局莫兰指数（Global Moran's I）
    
    参数:
        gdf: GeoDataFrame 对象
        field: 用于计算的数值字段
        weights: 空间权重类型
            - 'queen': 多边形相邻（共享边或点，近似用 touches）
            - 'distance': 距离带权重（需要 distance_threshold）
            - 'knn': k 近邻（基于质心距离，需要 k）
        distance_threshold: 距离带阈值（与 weights='distance' 搭配，单位与坐标一致）
        k: 近邻数量（与 weights='knn' 搭配）
        row_standardize: 是否进行行标准化（W 行和为 1）
        permutations: 置换检验次数（>= 0；0 表示不做置换检验）
        use_centroid: 对点位计算距离时是否使用几何质心（一般保持 True）
    
    返回:
        字典，包含：
            - I: 莫兰指数值
            - EI: 理论期望值 -1/(N-1)
            - z: 置换检验 z 值（若 permutations>0）
            - p_value: 置换检验 p 值（双尾，若 permutations>0）
            - S0: 权重总和
            - n: 样本数量
            - neighbors_stats: {"avg": 平均邻居数, "min": 最小, "max": 最大}
            - config: 输入配置回显
    
    说明:
        不依赖外部权重库，适用于小中等规模数据。大规模数据建议使用稀疏矩阵/专用库。
    """
    if field not in gdf.columns:
        raise ValueError(f"字段 '{field}' 不存在")
    if not pd.api.types.is_numeric_dtype(gdf[field]):
        raise ValueError("莫兰指数需要数值字段")
    if len(gdf) < 3:
        raise ValueError("样本量过小，至少需要3个要素")

    # 准备数值向量
    values = gdf[field].astype(float).to_numpy()
    mask = np.isfinite(values)
    if not mask.all():
        # 丢弃 NaN/Inf
        values = values[mask]
        gdf_local = gdf.loc[mask].reset_index(drop=True)
    else:
        gdf_local = gdf.reset_index(drop=True)

    n = len(values)
    x = values
    x_mean = x.mean()
    x_dev = x - x_mean
    ss = np.sum(x_dev ** 2)
    if ss == 0:
        raise ValueError("方差为0，无法计算莫兰指数")

    # 构建空间权重矩阵（稠密 N x N；大数据慎用）
    # 初始化为零
    W = np.zeros((n, n), dtype=float)

    # 质心坐标（用于距离类权重）
    if use_centroid:
        centroids = gdf_local.geometry.centroid
    else:
        centroids = gdf_local.geometry
    xs = centroids.x.to_numpy()
    ys = centroids.y.to_numpy()

    if weights == 'queen':
        # 多边形接触关系：近似使用 touches（边/点接触；对包含关系可视情况改为 intersects）
        geoms = gdf_local.geometry
        for i in range(n):
            gi = geoms.iloc[i]
            for j in range(i + 1, n):
                gj = geoms.iloc[j]
                try:
                    is_neighbor = gi.touches(gj)
                except Exception:
                    is_neighbor = False
                if is_neighbor:
                    W[i, j] = 1.0
                    W[j, i] = 1.0
    elif weights == 'distance':
        if distance_threshold is None or distance_threshold <= 0:
            raise ValueError("使用距离带权重需要正的 distance_threshold")
        # 距离带内 w_ij=1
        for i in range(n):
            dx = xs - xs[i]
            dy = ys - ys[i]
            d = np.hypot(dx, dy)
            neighbors = (d > 0) & (d <= float(distance_threshold))
            W[i, neighbors] = 1.0
    elif weights == 'knn':
        if k is None or k <= 0:
            raise ValueError("使用k近邻需要正整数 k")
        for i in range(n):
            dx = xs - xs[i]
            dy = ys - ys[i]
            d = np.hypot(dx, dy)
            order = np.argsort(d)
            # 排除自身（距离==0 的第一个）
            neighbors_idx = [idx for idx in order if idx != i][:int(k)]
            W[i, neighbors_idx] = 1.0
        # 对称化（无向图）
        W = np.maximum(W, W.T)
    else:
        raise ValueError("weights 仅支持 'queen' | 'distance' | 'knn'")

    # 行标准化（每行和=1），避免孤立点行和为0导致除零
    row_sums = W.sum(axis=1)
    if row_standardize:
        with np.errstate(invalid='ignore', divide='ignore'):
            W = np.divide(W, row_sums[:, None], out=np.zeros_like(W), where=row_sums[:, None] != 0)

    # 统计邻居数
    neighbors_count = (W > 0).sum(axis=1)
    neighbors_stats = {
        "avg": float(neighbors_count.mean()),
        "min": int(neighbors_count.min()),
        "max": int(neighbors_count.max())
    }

    # 权重总和 S0（行标准化后 S0≈N-孤立点数；未标准化时为边总数）
    S0 = W.sum()

    # 计算莫兰 I
    # I = (N / S0) * (x' W x) / (x' x)
    if S0 == 0:
        raise ValueError("权重总和 S0 为0，可能所有要素均无邻居，请调整权重参数")
    num = float(x_dev @ (W @ x_dev))
    I = (n / S0) * (num / ss)

    # 理论期望（正态近似）
    EI = -1.0 / (n - 1)

    result = {
        "I": float(I),
        "EI": float(EI),
        "S0": float(S0),
        "n": int(n),
        "neighbors_stats": neighbors_stats,
        "config": {
            "field": field,
            "weights": weights,
            "distance_threshold": distance_threshold,
            "k": int(k) if k is not None else None,
            "row_standardize": bool(row_standardize),
            "permutations": int(permutations)
        }
    }

    # 置换检验（双尾 p 值）
    if permutations and permutations > 0:
        rng = np.random.default_rng()
        perm_I = np.empty(permutations, dtype=float)
        for p in range(permutations):
            xp = rng.permutation(x)
            xp_dev = xp - xp.mean()
            perm_I[p] = (n / S0) * float(xp_dev @ (W @ xp_dev)) / float(np.sum(xp_dev ** 2))
        # 双尾 p：包含等于观测值的次数
        greater = np.sum(perm_I >= I)
        less = np.sum(perm_I <= I)
        p_value = (min(greater, less) + 1) / (permutations + 1)
        z = (I - perm_I.mean()) / (perm_I.std(ddof=1) + 1e-12)
        result.update({
            "z": float(z),
            "p_value": float(p_value)
        })

    return result


# ==================== 空间分析 ====================

def spatial_join(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, 
                how: str = 'inner', predicate: str = 'intersects') -> gpd.GeoDataFrame:
    """
    空间连接
    
    参数:
        gdf1: 第一个GeoDataFrame
        gdf2: 第二个GeoDataFrame
        how: 连接方式（'inner', 'left', 'right'）
        predicate: 空间关系
    
    返回:
        连接后的GeoDataFrame
    """
    # 确保坐标系一致
    if gdf1.crs != gdf2.crs and gdf2.crs is not None:
        gdf2 = gdf2.to_crs(gdf1.crs)
    
    return gpd.sjoin(gdf1, gdf2, how=how, predicate=predicate)


def calculate_distance_matrix(gdf1: gpd.GeoDataFrame, 
                              gdf2: Optional[gpd.GeoDataFrame] = None) -> np.ndarray:
    """
    计算距离矩阵
    
    参数:
        gdf1: 第一组要素
        gdf2: 第二组要素（如为None则计算gdf1内部距离）
    
    返回:
        距离矩阵
    """
    if gdf2 is None:
        gdf2 = gdf1
    else:
        # 确保坐标系一致
        if gdf1.crs != gdf2.crs and gdf2.crs is not None:
            gdf2 = gdf2.to_crs(gdf1.crs)
    
    n1, n2 = len(gdf1), len(gdf2)
    distances = np.zeros((n1, n2))
    
    for i, geom1 in enumerate(gdf1.geometry):
        for j, geom2 in enumerate(gdf2.geometry):
            distances[i, j] = geom1.distance(geom2)
    
    return distances


def find_nearest_features(gdf: gpd.GeoDataFrame, 
                         reference_gdf: gpd.GeoDataFrame, 
                         k: int = 1) -> gpd.GeoDataFrame:
    """
    查找最近的k个要素
    
    参数:
        gdf: 要查询的GeoDataFrame
        reference_gdf: 参考GeoDataFrame
        k: 查找数量
    
    返回:
        包含最近要素信息的GeoDataFrame
    """
    # 确保坐标系一致
    if gdf.crs != reference_gdf.crs and reference_gdf.crs is not None:
        reference_gdf = reference_gdf.to_crs(gdf.crs)
    
    result = gdf.copy()
    nearest_indices = []
    nearest_distances = []
    
    for geom in gdf.geometry:
        distances = reference_gdf.geometry.distance(geom)
        nearest_idx = distances.nsmallest(k).index.tolist()
        nearest_dist = distances.nsmallest(k).values.tolist()
        nearest_indices.append(nearest_idx)
        nearest_distances.append(nearest_dist)
    
    result['nearest_indices'] = nearest_indices
    result['nearest_distances'] = nearest_distances
    
    return result


# ==================== 数据验证 ====================

def check_geometry_validity(gdf: gpd.GeoDataFrame) -> Tuple[bool, List[int]]:
    """
    检查几何有效性
    
    返回:
        (是否全部有效, 无效要素索引列表)
    """
    invalid_indices = []
    
    for idx, geom in enumerate(gdf.geometry):
        if not geom.is_valid:
            invalid_indices.append(idx)
    
    is_all_valid = len(invalid_indices) == 0
    return is_all_valid, invalid_indices


def fix_invalid_geometries(gdf: gpd.GeoDataFrame, inplace: bool = False) -> gpd.GeoDataFrame:
    """
    修复无效几何
    
    参数:
        gdf: GeoDataFrame对象
        inplace: 是否就地修改
    
    返回:
        修复后的GeoDataFrame
    """
    if not inplace:
        gdf = gdf.copy()
    
    gdf.geometry = gdf.geometry.buffer(0)
    return gdf


def remove_duplicate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    删除重复几何
    
    返回:
        删除重复后的GeoDataFrame
    """
    return gdf.drop_duplicates(subset=['geometry'])


# ==================== 实用工具 ====================

def print_stats_summary(gdf: gpd.GeoDataFrame):
    """
    打印统计摘要
    """
    print("="*60)
    print("Shapefile统计摘要")
    print("="*60)
    
    # 基本信息
    basic = get_basic_stats(gdf)
    print(f"\n要素数量: {basic['要素数量']}")
    print(f"字段数量: {basic['字段数量']}")
    print(f"坐标系: {basic['坐标系']}")
    print(f"几何类型: {basic['几何类型']}")
    
    # 几何统计
    geom_stats = get_geometry_stats(gdf)
    print(f"\n几何统计:")
    for key, value in geom_stats.items():
        if key != '几何类型':
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("="*60)


def export_to_csv(gdf: gpd.GeoDataFrame, output_path: str, 
                 include_geometry: bool = False) -> bool:
    """
    导出属性表到CSV
    
    参数:
        gdf: GeoDataFrame对象
        output_path: 输出路径
        include_geometry: 是否包含几何字段（WKT格式）
    
    返回:
        是否成功
    """
    try:
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        if include_geometry:
            df['geometry_wkt'] = gdf.geometry.to_wkt()
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        print(f"导出CSV失败: {e}")
        return False


def get_field_summary(gdf: gpd.GeoDataFrame, field: str) -> Dict:
    """
    获取单个字段的详细统计摘要
    
    参数:
        gdf: GeoDataFrame对象
        field: 字段名
    
    返回:
        统计摘要字典
    
    示例:
        summary = get_field_summary(gdf, 'population')
    """
    if field not in gdf.columns:
        raise ValueError(f"字段 '{field}' 不存在")
    
    summary = {
        "字段名": field,
        "数据类型": str(gdf[field].dtype),
        "总数": len(gdf[field]),
        "非空值": int(gdf[field].count()),
        "空值": int(gdf[field].isna().sum()),
        "空值比例": f"{gdf[field].isna().sum() / len(gdf) * 100:.2f}%"
    }
    
    if pd.api.types.is_numeric_dtype(gdf[field]):
        summary.update({
            "最小值": float(gdf[field].min()),
            "最大值": float(gdf[field].max()),
            "平均值": float(gdf[field].mean()),
            "中位数": float(gdf[field].median()),
            "标准差": float(gdf[field].std()),
            "四分位数": {
                "25%": float(gdf[field].quantile(0.25)),
                "50%": float(gdf[field].quantile(0.50)),
                "75%": float(gdf[field].quantile(0.75))
            }
        })
    else:
        unique_count = gdf[field].nunique()
        summary.update({
            "唯一值数量": int(unique_count),
            "最常见值": gdf[field].mode()[0] if len(gdf[field].mode()) > 0 else None,
            "最常见值频数": int(gdf[field].value_counts().iloc[0]) if len(gdf[field].value_counts()) > 0 else 0
        })
        
        if unique_count <= 20:
            summary["唯一值列表"] = get_unique_values(gdf, field)
    
    return summary


def group_statistics(gdf: gpd.GeoDataFrame, group_field: str, 
                    stat_field: str, agg_funcs: List[str] = None) -> pd.DataFrame:
    """
    按字段分组统计
    
    参数:
        gdf: GeoDataFrame对象
        group_field: 分组字段
        stat_field: 统计字段
        agg_funcs: 聚合函数列表（默认为['count', 'sum', 'mean', 'min', 'max']）
    
    返回:
        统计结果DataFrame
    
    示例:
        stats = group_statistics(gdf, 'category', 'population')
        stats = group_statistics(gdf, 'district', 'area', ['sum', 'mean', 'std'])
    """
    if agg_funcs is None:
        agg_funcs = ['count', 'sum', 'mean', 'min', 'max']
    
    grouped = gdf.groupby(group_field)[stat_field].agg(agg_funcs)
    return grouped


def count_by_field(gdf: gpd.GeoDataFrame, field: str, 
                  sort: bool = True, ascending: bool = False) -> pd.Series:
    """
    统计字段值的频数
    
    参数:
        gdf: GeoDataFrame对象
        field: 字段名
        sort: 是否排序
        ascending: 是否升序
    
    返回:
        频数统计Series
    
    示例:
        counts = count_by_field(gdf, 'category')
        print(counts)
    """
    counts = gdf[field].value_counts(sort=sort, ascending=ascending)
    return counts


def sample_features(gdf: gpd.GeoDataFrame, n: int = None, frac: float = None, 
                   random_state: int = None) -> gpd.GeoDataFrame:
    """
    随机抽样要素
    
    参数:
        gdf: GeoDataFrame对象
        n: 抽样数量
        frac: 抽样比例（0-1之间）
        random_state: 随机种子
    
    返回:
        抽样后的GeoDataFrame
    
    示例:
        sample = sample_features(gdf, n=100)  # 抽取100个要素
        sample = sample_features(gdf, frac=0.1)  # 抽取10%的要素
    """
    return gdf.sample(n=n, frac=frac, random_state=random_state)


def merge_by_field(gdf: gpd.GeoDataFrame, df: pd.DataFrame, 
                  on: str = None, left_on: str = None, right_on: str = None,
                  how: str = 'left') -> gpd.GeoDataFrame:
    """
    根据字段合并属性表
    
    参数:
        gdf: GeoDataFrame对象
        df: 要合并的DataFrame
        on: 连接字段（两边相同）
        left_on: 左侧（gdf）连接字段
        right_on: 右侧（df）连接字段
        how: 连接方式（'left', 'right', 'inner', 'outer'）
    
    返回:
        合并后的GeoDataFrame
    
    示例:
        # 使用相同字段名连接
        merged = merge_by_field(gdf, attr_df, on='id')
        
        # 使用不同字段名连接
        merged = merge_by_field(gdf, attr_df, left_on='id', right_on='feature_id')
    """
    if on is not None:
        result = gdf.merge(df, on=on, how=how)
    else:
        result = gdf.merge(df, left_on=left_on, right_on=right_on, how=how)
    
    return gpd.GeoDataFrame(result, geometry='geometry', crs=gdf.crs)


def get_file_info(shp_path: str) -> Dict:
    """
    获取shp文件信息（不完全加载数据）
    
    返回:
        文件信息字典
    """
    import fiona
    
    with fiona.open(shp_path) as src:
        info = {
            "文件路径": shp_path,
            "要素数量": len(src),
            "坐标系": str(src.crs),
            "边界范围": src.bounds,
            "几何类型": src.schema['geometry'],
            "字段信息": src.schema['properties']
        }
    
    return info


# ==================== 功能性分析 ====================

def shannon_entropy(values: np.ndarray) -> float:
    """
    计算香农熵
    
    参数:
        values: 比例值数组（和为1）
    
    返回:
        香农熵值
    """
    values = values[values > 0]
    return float(-np.sum(values * np.log(values)))


def simpson_index(values: np.ndarray) -> float:
    """
    计算辛普森指数
    
    参数:
        values: 比例值数组（和为1）
    
    返回:
        辛普森指数值
    """
    return float(1 - np.sum(values ** 2))


def functional_diversity_analysis(gdf: gpd.GeoDataFrame, 
                                  label_field: str = 'label',
                                  label_names: Dict = None,
                                  verbose: bool = True) -> Dict:
    """
    功能多样性分析
    
    参数:
        gdf: GeoDataFrame对象
        label_field: 标签字段名
        label_names: 标签映射字典 {标签值: 标签名称}
        verbose: 是否打印详细信息
    
    返回:
        包含各项指标的字典
    
    示例:
        label_names = {1: "商业", 2: "住宅", 3: "公共服务"}
        results = functional_diversity_analysis(gdf, 'label', label_names)
    """
    if label_field not in gdf.columns:
        raise ValueError(f"字段 '{label_field}' 不存在")
    
    total = len(gdf)
    label_counts = count_by_field(gdf, label_field)
    label_proportions = label_counts / total
    values = label_proportions.values
    
    # 计算指标
    entropy = shannon_entropy(values)
    simpson = simpson_index(values)
    n_categories = len(label_counts)
    functional_balance = entropy / np.log(n_categories) if n_categories > 1 else 0.0
    
    results = {
        'total_features': total,
        'n_categories': n_categories,
        'label_counts': label_counts.to_dict(),
        'label_proportions': label_proportions.to_dict(),
        'shannon_entropy': entropy,
        'simpson_index': simpson,
        'functional_balance': functional_balance
    }
    
    if verbose:
        print("="*60)
        print("功能多样性分析")
        print("="*60)
        print(f"\n总要素数: {total}")
        print(f"功能类别数: {n_categories}")
        
        print(f"\n各类别分布:")
        for label, count in label_counts.items():
            name = label_names.get(label, str(label)) if label_names else str(label)
            pct = count / total * 100
            print(f"  {name}: {count} ({pct:.2f}%)")
        
        print(f"\n多样性指标:")
        print(f"  香农熵: {entropy:.4f}")
        print(f"  辛普森指数: {simpson:.4f}")
        print(f"  功能均衡度: {functional_balance:.4f} (0-1, 越接近1越均衡)")
    
    return results


def compute_morans_i_by_label(gdf: gpd.GeoDataFrame,
                              label_field: str = 'label',
                              label_names: Dict = None,
                              weights: str = 'knn',
                              k: int = 8,
                              distance_threshold: float = None,
                              permutations: int = 999,
                              verbose: bool = True) -> Dict:
    """
    计算各标签的全局莫兰指数
    
    参数:
        gdf: GeoDataFrame对象
        label_field: 标签字段名
        label_names: 标签映射字典
        weights: 空间权重类型 ('knn', 'queen', 'distance')
        k: k近邻数量
        distance_threshold: 距离阈值
        permutations: 置换检验次数
        verbose: 是否打印详细信息
    
    返回:
        各标签的莫兰指数结果字典
    
    示例:
        results = compute_morans_i_by_label(gdf, 'label', weights='knn', k=8)
    """
    if label_field not in gdf.columns:
        raise ValueError(f"字段 '{label_field}' 不存在")
    
    unique_labels = sorted(gdf[label_field].unique())
    morans_results = {}
    
    if verbose:
        print("\n" + "="*60)
        print("空间自相关分析（全局莫兰指数）")
        print("="*60)
    
    for label in unique_labels:
        label_name = label_names.get(label, str(label)) if label_names else str(label)
        binary_field = f'_binary_{label}'
        gdf[binary_field] = (gdf[label_field] == label).astype(int)
        
        try:
            result = compute_morans_i(
                gdf, field=binary_field, weights=weights,
                distance_threshold=distance_threshold, k=k,
                row_standardize=True, permutations=permutations,
                use_centroid=True
            )
            
            morans_results[label] = result
            
            if verbose:
                print(f"\n【{label_name}】")
                print(f"  莫兰指数 I: {result['I']:.4f}")
                print(f"  期望值 E(I): {result['EI']:.4f}")
                
                if 'z' in result and 'p_value' in result:
                    print(f"  Z值: {result['z']:.4f}")
                    print(f"  P值: {result['p_value']:.4f}")
                    
                    # 显著性
                    if result['p_value'] < 0.01:
                        sig = "***"
                    elif result['p_value'] < 0.05:
                        sig = "**"
                    elif result['p_value'] < 0.1:
                        sig = "*"
                    else:
                        sig = "n.s."
                    print(f"  显著性: {sig}")
                    
                    # 空间模式
                    if result['I'] > result['EI'] and result['p_value'] < 0.05:
                        pattern = "空间聚集"
                    elif result['I'] < result['EI'] and result['p_value'] < 0.05:
                        pattern = "空间离散"
                    else:
                        pattern = "随机分布"
                    print(f"  空间模式: {pattern}")
        
        except Exception as e:
            if verbose:
                print(f"\n【{label_name}】计算失败: {e}")
            morans_results[label] = None
        
        # 清理临时字段
        gdf.drop(columns=[binary_field], inplace=True)
    
    return morans_results


def compute_getis_ord_gi_star(gdf: gpd.GeoDataFrame,
                               field: str,
                               weights: str = 'knn',
                               k: int = 8,
                               distance_threshold: float = None) -> pd.DataFrame:
    """
    计算Getis-Ord Gi*统计量（热点分析）
    
    参数:
        gdf: GeoDataFrame对象
        field: 分析字段
        weights: 权重类型 ('knn', 'distance')
        k: k近邻数量
        distance_threshold: 距离阈值
    
    返回:
        包含Gi*、Z分数、P值和热点类型的DataFrame
    
    示例:
        hotspots = compute_getis_ord_gi_star(gdf, 'population', weights='knn', k=8)
    """
    from scipy import stats
    
    n = len(gdf)
    values = gdf[field].astype(float).to_numpy()
    
    # 计算质心
    centroids = gdf.geometry.centroid
    xs = centroids.x.to_numpy()
    ys = centroids.y.to_numpy()
    
    # 构建权重矩阵
    W = np.zeros((n, n), dtype=float)
    
    if weights == 'knn':
        for i in range(n):
            dx = xs - xs[i]
            dy = ys - ys[i]
            d = np.hypot(dx, dy)
            order = np.argsort(d)
            neighbors_idx = [idx for idx in order if idx != i][:int(k)]
            W[i, neighbors_idx] = 1.0
    elif weights == 'distance':
        if distance_threshold is None or distance_threshold <= 0:
            raise ValueError("使用距离带权重需要正的distance_threshold")
        for i in range(n):
            dx = xs - xs[i]
            dy = ys - ys[i]
            d = np.hypot(dx, dy)
            neighbors = (d > 0) & (d <= float(distance_threshold))
            W[i, neighbors] = 1.0
    else:
        raise ValueError("weights仅支持'knn'或'distance'")
    
    # 计算Gi*
    gi_star = np.zeros(n)
    z_scores = np.zeros(n)
    p_values = np.zeros(n)
    
    x_mean = values.mean()
    x_std = values.std()
    
    for i in range(n):
        w_star = W[i].copy()
        w_star[i] = 1.0  # 包含自身
        
        weighted_sum = np.sum(w_star * values)
        sum_w = np.sum(w_star)
        sum_w2 = np.sum(w_star ** 2)
        
        E_Gi = x_mean * sum_w
        term1 = sum_w2 * (n - sum_w)
        term2 = (n - 1) * sum_w ** 2
        Var_Gi = x_std ** 2 * (term1 - term2) / (n - 1)
        
        if Var_Gi > 0:
            SD_Gi = np.sqrt(Var_Gi)
            z_scores[i] = (weighted_sum - E_Gi) / SD_Gi
            p_values[i] = 2 * (1 - stats.norm.cdf(abs(z_scores[i])))
        else:
            z_scores[i] = 0
            p_values[i] = 1.0
        
        gi_star[i] = weighted_sum
    

    # 分类热点类型（修正版 - 修复阈值错误）
    hotspot_type = np.where(
    (z_scores > 2.58) & (p_values < 0.01), 'Hot Spot (99%)',      # 99%置信水平
    np.where((z_scores > 1.96) & (p_values < 0.05), 'Hot Spot (95%)',  # 95%置信水平  
    np.where((z_scores > 1.65) & (p_values < 0.10), 'Hot Spot (90%)',  # 90%置信水平
    np.where((z_scores < -2.58) & (p_values < 0.01), 'Cold Spot (99%)',
    np.where((z_scores < -1.96) & (p_values < 0.05), 'Cold Spot (95%)',
    np.where((z_scores < -1.65) & (p_values < 0.10), 'Cold Spot (90%)',
    'Not Significant'))))))
    
    return pd.DataFrame({
        'Gi_star': gi_star,
        'Z_score': z_scores,
        'P_value': p_values,
        'Hotspot_Type': hotspot_type
    })


def hotspot_analysis_by_label(gdf: gpd.GeoDataFrame,
                              label_field: str = 'label',
                              label_names: Dict = None,
                              weights: str = 'knn',
                              k: int = 8,
                              distance_threshold: float = None,
                              add_to_gdf: bool = True,
                              verbose: bool = True) -> Tuple[Dict, gpd.GeoDataFrame]:
    """
    对各标签进行热点分析
    
    参数:
        gdf: GeoDataFrame对象
        label_field: 标签字段名
        label_names: 标签映射字典
        weights: 权重类型
        k: k近邻数量
        distance_threshold: 距离阈值
        add_to_gdf: 是否将结果添加到GeoDataFrame
        verbose: 是否打印详细信息
    
    返回:
        (热点分析结果字典, 更新后的GeoDataFrame)
    
    示例:
        results, gdf = hotspot_analysis_by_label(gdf, 'label', weights='knn', k=8)
    """
    if label_field not in gdf.columns:
        raise ValueError(f"字段 '{label_field}' 不存在")
    
    if add_to_gdf:
        gdf = gdf.copy()
    
    unique_labels = sorted(gdf[label_field].unique())
    hotspot_results = {}
    
    if verbose:
        print("\n" + "="*60)
        print("热点分析（Getis-Ord Gi*）")
        print("="*60)
    
    for label in unique_labels:
        label_name = label_names.get(label, str(label)) if label_names else str(label)
        binary_field = f'_binary_{label}'
        gdf[binary_field] = (gdf[label_field] == label).astype(int)
        
        try:
            if verbose:
                print(f"\n分析【{label_name}】...")
            
            gi_results = compute_getis_ord_gi_star(
                gdf, field=binary_field, weights=weights,
                k=k, distance_threshold=distance_threshold
            )
            
            if add_to_gdf:
                gdf[f'{label}_GiStar'] = gi_results['Gi_star']
                gdf[f'{label}_Zscore'] = gi_results['Z_score']
                gdf[f'{label}_Pvalue'] = gi_results['P_value']
                gdf[f'{label}_Hotspot'] = gi_results['Hotspot_Type']
            
            hotspot_results[label] = gi_results
            
            if verbose:
                counts = gi_results['Hotspot_Type'].value_counts()
                for htype, count in counts.items():
                    pct = count / len(gdf) * 100
                    print(f"  {htype}: {count} ({pct:.2f}%)")
        
        except Exception as e:
            if verbose:
                print(f"  失败: {e}")
            hotspot_results[label] = None
        
        # 清理临时字段
        gdf.drop(columns=[binary_field], inplace=True)
    
    return hotspot_results, gdf


# ==================== 加速版本（多线程/多进程） ====================

def _compute_morans_single_label(args):
    """
    计算单个标签的莫兰指数（用于并行处理）
    
    内部函数，不直接调用
    """
    gdf, label, label_field, label_name, weights, k, distance_threshold, permutations = args
    
    binary_field = f'_binary_{label}'
    gdf_copy = gdf.copy()
    gdf_copy[binary_field] = (gdf_copy[label_field] == label).astype(int)
    
    try:
        result = compute_morans_i(
            gdf_copy, field=binary_field, weights=weights,
            distance_threshold=distance_threshold, k=k,
            row_standardize=True, permutations=permutations,
            use_centroid=True
        )
        return (label, label_name, result, None)
    except Exception as e:
        return (label, label_name, None, str(e))


def compute_morans_i_by_label_parallel(gdf: gpd.GeoDataFrame,
                                       label_field: str = 'label',
                                       label_names: Dict = None,
                                       weights: str = 'knn',
                                       k: int = 8,
                                       distance_threshold: float = None,
                                       permutations: int = 999,
                                       n_jobs: int = -1,
                                       verbose: bool = True) -> Dict:
    """
    计算各标签的全局莫兰指数（并行加速版）
    
    参数:
        gdf: GeoDataFrame对象
        label_field: 标签字段名
        label_names: 标签映射字典
        weights: 空间权重类型
        k: k近邻数量
        distance_threshold: 距离阈值
        permutations: 置换检验次数
        n_jobs: 并行进程数（-1表示使用所有CPU核心）
        verbose: 是否打印详细信息
    
    返回:
        各标签的莫兰指数结果字典
    
    示例:
        results = compute_morans_i_by_label_parallel(gdf, n_jobs=4)
    """
    from multiprocessing import Pool, cpu_count
    
    if label_field not in gdf.columns:
        raise ValueError(f"字段 '{label_field}' 不存在")
    
    unique_labels = sorted(gdf[label_field].unique())
    
    if verbose:
        print("\n" + "="*60)
        print("空间自相关分析（全局莫兰指数 - 并行加速）")
        print("="*60)
        n_cores = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
        print(f"使用 {n_cores} 个CPU核心并行计算")
    
    # 准备参数
    tasks = []
    for label in unique_labels:
        label_name = label_names.get(label, str(label)) if label_names else str(label)
        tasks.append((gdf, label, label_field, label_name, weights, k, 
                     distance_threshold, permutations))
    
    # 并行计算
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    with Pool(processes=n_jobs) as pool:
        results_list = pool.map(_compute_morans_single_label, tasks)
    
    # 整理结果
    morans_results = {}
    for label, label_name, result, error in results_list:
        morans_results[label] = result
        
        if verbose:
            print(f"\n【{label_name}】")
            if result is not None:
                print(f"  莫兰指数 I: {result['I']:.4f}")
                print(f"  期望值 E(I): {result['EI']:.4f}")
                
                if 'z' in result and 'p_value' in result:
                    print(f"  Z值: {result['z']:.4f}")
                    print(f"  P值: {result['p_value']:.4f}")
                    
                    if result['p_value'] < 0.01:
                        sig = "***"
                    elif result['p_value'] < 0.05:
                        sig = "**"
                    elif result['p_value'] < 0.1:
                        sig = "*"
                    else:
                        sig = "n.s."
                    print(f"  显著性: {sig}")
                    
                    if result['I'] > result['EI'] and result['p_value'] < 0.05:
                        pattern = "空间聚集"
                    elif result['I'] < result['EI'] and result['p_value'] < 0.05:
                        pattern = "空间离散"
                    else:
                        pattern = "随机分布"
                    print(f"  空间模式: {pattern}")
            else:
                print(f"  计算失败: {error}")
    
    return morans_results



# ==================== GPU加速版本（可选） ====================

def compute_morans_i_gpu(gdf: gpd.GeoDataFrame,
                        field: str,
                        weights: str = 'knn',
                        k: int = 8,
                        distance_threshold: float = None,
                        row_standardize: bool = True,
                        use_centroid: bool = True) -> Dict:
    """
    计算全局莫兰指数（GPU加速版）
    
    需要安装: pip install cupy-cuda11x  (根据CUDA版本选择)
    
    参数与compute_morans_i相同
    
    返回:
        莫兰指数结果字典
    
    注意:
        - 需要NVIDIA GPU和CUDA
        - 适合大规模数据集（>10000个要素）
        - 小数据集可能不如CPU版快
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError("GPU加速需要安装CuPy: pip install cupy-cuda11x")
    
    if field not in gdf.columns:
        raise ValueError(f"字段 '{field}' 不存在")
    if not pd.api.types.is_numeric_dtype(gdf[field]):
        raise ValueError("莫兰指数需要数值字段")
    if len(gdf) < 3:
        raise ValueError("样本量过小，至少需要3个要素")
    
    # 准备数据
    values = gdf[field].astype(float).to_numpy()
    mask = np.isfinite(values)
    if not mask.all():
        values = values[mask]
        gdf_local = gdf.loc[mask].reset_index(drop=True)
    else:
        gdf_local = gdf.reset_index(drop=True)
    
    n = len(values)
    
    # 转移到GPU
    x_gpu = cp.array(values, dtype=cp.float32)
    x_mean = float(cp.mean(x_gpu))
    x_dev = x_gpu - x_mean
    ss = float(cp.sum(x_dev ** 2))
    
    if ss == 0:
        raise ValueError("方差为0，无法计算莫兰指数")
    
    # 计算质心坐标
    if use_centroid:
        centroids = gdf_local.geometry.centroid
    else:
        centroids = gdf_local.geometry
    
    xs = cp.array(centroids.x.to_numpy(), dtype=cp.float32)
    ys = cp.array(centroids.y.to_numpy(), dtype=cp.float32)
    
    # 构建权重矩阵（GPU加速）
    W_gpu = cp.zeros((n, n), dtype=cp.float32)
    
    if weights == 'knn':
        # 计算距离矩阵
        xs_2d = xs.reshape(-1, 1)
        ys_2d = ys.reshape(-1, 1)
        dx = xs_2d - xs
        dy = ys_2d - ys
        dist_matrix = cp.sqrt(dx**2 + dy**2)
        
        # k近邻
        for i in range(n):
            distances = dist_matrix[i]
            # 找到k个最近邻（排除自己）
            k_nearest_idx = cp.argsort(distances)[1:k+1]
            W_gpu[i, k_nearest_idx] = 1.0
        
        # 对称化
        W_gpu = cp.maximum(W_gpu, W_gpu.T)
        
    elif weights == 'distance':
        if distance_threshold is None or distance_threshold <= 0:
            raise ValueError("使用距离带权重需要正的distance_threshold")
        
        xs_2d = xs.reshape(-1, 1)
        ys_2d = ys.reshape(-1, 1)
        dx = xs_2d - xs
        dy = ys_2d - ys
        dist_matrix = cp.sqrt(dx**2 + dy**2)
        
        W_gpu = ((dist_matrix > 0) & (dist_matrix <= distance_threshold)).astype(cp.float32)
    
    else:
        raise ValueError("GPU版本仅支持'knn'和'distance'权重")
    
    # 行标准化
    if row_standardize:
        row_sums = cp.sum(W_gpu, axis=1, keepdims=True)
        W_gpu = cp.where(row_sums != 0, W_gpu / row_sums, 0)
    
    # 计算莫兰指数
    S0 = float(cp.sum(W_gpu))
    if S0 == 0:
        raise ValueError("权重总和S0为0")
    
    num = float(cp.dot(x_dev, cp.dot(W_gpu, x_dev)))
    I = (n / S0) * (num / ss)
    EI = -1.0 / (n - 1)
    
    # 邻居统计
    neighbors_count = cp.sum(W_gpu > 0, axis=1)
    neighbors_stats = {
        "avg": float(cp.mean(neighbors_count)),
        "min": int(cp.min(neighbors_count)),
        "max": int(cp.max(neighbors_count))
    }
    
    result = {
        "I": float(I),
        "EI": float(EI),
        "S0": float(S0),
        "n": int(n),
        "neighbors_stats": neighbors_stats,
        "config": {
            "field": field,
            "weights": weights,
            "distance_threshold": distance_threshold,
            "k": int(k) if k is not None else None,
            "row_standardize": bool(row_standardize),
            "accelerator": "GPU"
        }
    }
    
    return result

from scipy import stats
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Optional, Union
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_categorical_hotspot(gdf: gpd.GeoDataFrame,
                               label_field: str,
                               label_names: Optional[Dict] = None,
                               weights: str = 'knn',
                               k: int = 8,
                               distance_threshold: Optional[float] = None,
                               method: str = 'binary',
                               auto_fix_projection: bool = True,
                               add_to_gdf: bool = True,
                               verbose: bool = True) -> Dict:
    """
    对分类数据进行热点分析
    
    参数:
        gdf: GeoDataFrame对象
        label_field: 标签字段名（如'land_use', 'category'等）
        label_names: 标签名称映射字典，如{1: "商业", 2: "住宅"}
        weights: 权重类型 ('knn'=k近邻, 'distance'=距离带)
        k: k近邻数量
        distance_threshold: 距离阈值
        method: 分析方法
            - 'binary': 二值化方法（推荐）- 每个类别转为0/1
            - 'density': 密度方法 - 计算局部比例
            - 'count': 计数方法 - 邻域内该类别的数量
        auto_fix_projection: 自动修复投影问题
        add_to_gdf: 是否将结果添加到GeoDataFrame
        verbose: 是否输出详细信息
    
    返回:
        各类别的热点分析结果字典
    """
    
    if verbose:
        print("="*60)
        print("分类数据热点分析（Categorical Getis-Ord Gi*）")
        print("="*60)
    
    # 1. 数据验证
    if label_field not in gdf.columns:
        raise ValueError(f"字段 '{label_field}' 不存在")
    
    # 2. 处理投影
    gdf_work = gdf.copy()
    
    if auto_fix_projection:
        if gdf_work.crs is None:
            if verbose:
                print("警告：没有定义坐标系，设置为WGS84")
            gdf_work = gdf_work.set_crs('EPSG:4326')
        
        if gdf_work.crs.is_geographic:
            if verbose:
                print("警告：使用地理坐标系，转换到Web Mercator投影...")
            gdf_work = gdf_work.to_crs('EPSG:3857')
    
    # 3. 获取唯一标签
    unique_labels = sorted(gdf_work[label_field].dropna().unique())
    
    if verbose:
        print(f"\n发现 {len(unique_labels)} 个类别:")
        for label in unique_labels:
            name = label_names.get(label, str(label)) if label_names else str(label)
            count = (gdf_work[label_field] == label).sum()
            pct = count / len(gdf_work) * 100
            print(f"  {label}: {name} - {count}个 ({pct:.1f}%)")
    
    # 4. 构建空间权重矩阵
    if verbose:
        print(f"\n构建空间权重矩阵 (weights={weights}, k={k})...")
    
    n = len(gdf_work)
    
    # 计算质心
    centroids = gdf_work.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])
    
    # 构建距离矩阵
    dist_matrix = distance_matrix(coords, coords)
    
    # 构建权重矩阵
    W = np.zeros((n, n))
    
    if weights == 'knn':
        for i in range(n):
            distances = dist_matrix[i].copy()
            distances[i] = np.inf
            k_actual = min(k, n-1)
            nearest_indices = np.argpartition(distances, k_actual)[:k_actual]
            W[i, nearest_indices] = 1.0
        W = np.maximum(W, W.T)  # 对称化
        
    elif weights == 'distance':
        if distance_threshold is None:
            min_distances = []
            for i in range(n):
                distances = dist_matrix[i].copy()
                distances[i] = np.inf
                min_distances.append(distances.min())
            distance_threshold = np.mean(min_distances) * 3
            if verbose:
                print(f"自动设置距离阈值: {distance_threshold:.2f}")
        
        W = (dist_matrix > 0) & (dist_matrix <= distance_threshold)
        W = W.astype(float)
    
    # 检查孤立要素
    row_sums = W.sum(axis=1)
    isolated = row_sums == 0
    if isolated.any():
        isolated_count = isolated.sum()
        if verbose:
            print(f"警告：{isolated_count} 个要素没有邻居，为其添加最近邻")
        for i in np.where(isolated)[0]:
            distances = dist_matrix[i].copy()
            distances[i] = np.inf
            nearest_idx = np.argmin(distances)
            W[i, nearest_idx] = 1.0
            W[nearest_idx, i] = 1.0
    
    # 行标准化
    row_sums = W.sum(axis=1, keepdims=True)
    W = np.divide(W, row_sums, where=row_sums!=0)
    
    # 5. 对每个类别进行热点分析
    results = {}
    
    for label in unique_labels:
        label_name = label_names.get(label, str(label)) if label_names else str(label)
        
        if verbose:
            print(f"\n分析类别: {label_name}")
            print("-"*40)
        
        # 创建分析变量
        if method == 'binary':
            # 二值化：该类别为1，其他为0
            values = (gdf_work[label_field] == label).astype(float).values
            
        elif method == 'density':
            # 密度：计算邻域内该类别的比例
            values = (gdf_work[label_field] == label).astype(float).values
            # 这里仍使用二值，但解释为密度
            
        elif method == 'count':
            # 计数：简单计数
            values = (gdf_work[label_field] == label).astype(float).values
        
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        # 计算统计量
        n_category = values.sum()
        
        if n_category == 0:
            if verbose:
                print(f"  跳过：没有该类别的要素")
            continue
        
        if n_category == n:
            if verbose:
                print(f"  跳过：所有要素都属于该类别")
            continue
        
        # 计算全局统计
        x_mean = np.mean(values)
        var_x = np.var(values, ddof=1)
        
        if var_x == 0:
            if verbose:
                print(f"  跳过：方差为0")
            continue
        
        S = np.sqrt(var_x)
        
        # 计算Gi*统计量
        Gi_star = np.zeros(n)
        Z_scores = np.zeros(n)
        P_values = np.ones(n)
        
        for i in range(n):
            # 包含自身的权重
            W_i = W[i].copy()
            W_i[i] = 1.0  # Gi*包含自身
            
            # 计算加权和
            numerator = np.sum(W_i * values) - x_mean * np.sum(W_i)
            
            # 计算标准差
            n_wi = np.sum(W_i)
            n_wi_sq = np.sum(W_i ** 2)
            
            denominator_sq = ((n * n_wi_sq - n_wi ** 2) / (n - 1))
            
            if denominator_sq <= 0:
                continue
            
            denominator = S * np.sqrt(denominator_sq)
            
            if denominator > 0:
                Gi_star[i] = numerator / denominator
                Z_scores[i] = Gi_star[i]
                P_values[i] = 2 * (1 - stats.norm.cdf(abs(Z_scores[i])))
        
        # 分类热点类型
        hotspot_types = []
        for z, p in zip(Z_scores, P_values):
            if p <= 0.01:
                if z > 0:
                    hotspot_types.append(f'{label_name}-热点-99%')
                else:
                    hotspot_types.append(f'{label_name}-冷点-99%')
            elif p <= 0.05:
                if z > 0:
                    hotspot_types.append(f'{label_name}-热点-95%')
                else:
                    hotspot_types.append(f'{label_name}-冷点-95%')
            elif p <= 0.1:
                if z > 0:
                    hotspot_types.append(f'{label_name}-热点-90%')
                else:
                    hotspot_types.append(f'{label_name}-冷点-90%')
            else:
                hotspot_types.append('不显著')
        
        # 保存结果
        result = {
            'label': label,
            'label_name': label_name,
            'n_features': int(n_category),
            'proportion': float(n_category / n),
            'Gi_star': Gi_star,
            'Z_score': Z_scores,
            'P_value': P_values,
            'Hotspot_Type': pd.Series(hotspot_types)
        }
        
        results[label] = result
        
        # 添加到GeoDataFrame
        if add_to_gdf:
            gdf[f'{label_field}_{label}_Z'] = Z_scores
            gdf[f'{label_field}_{label}_P'] = P_values
            gdf[f'{label_field}_{label}_Type'] = hotspot_types
        
        # 输出统计
        if verbose:
            print(f"  要素数量: {n_category} ({n_category/n*100:.1f}%)")
            
            # 统计热点和冷点数量
            n_hot_sig = sum(1 for t in hotspot_types if '热点' in str(t))
            n_cold_sig = sum(1 for t in hotspot_types if '冷点' in str(t))
            n_not_sig = sum(1 for t in hotspot_types if t == '不显著')
            
            print(f"  热点区域: {n_hot_sig} ({n_hot_sig/n*100:.1f}%)")
            #print(f"  冷点区域: {n_cold_sig} ({n_cold_sig/n*100:.1f}%)")
            print(f"  不显著: {n_not_sig} ({n_not_sig/n*100:.1f}%)")
    
    if verbose:
        print("\n" + "="*60)
        print("热点分析完成")
        print("="*60)
    
    return results


def visualize_categorical_hotspots(gdf: gpd.GeoDataFrame,
                                   label_field: str,
                                   results: Dict,
                                   label_names: Optional[Dict] = None,
                                   figsize: tuple = (15, 10),
                                   save_path: Optional[str] = None):
    """
    可视化分类热点分析结果
    
    参数:
        gdf: GeoDataFrame对象
        label_field: 标签字段名
        results: 热点分析结果
        label_names: 标签名称映射字典
        figsize: 图形大小
        save_path: 保存路径
    """
    
    n_labels = len(results)
    
    if n_labels == 0:
        print("没有结果可视化")
        return
    
    # 计算子图布局
    cols = min(3, n_labels)
    rows = (n_labels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_labels == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 定义颜色方案
    colors = {
        '热点-99%': '#8B0000',  # 深红
        '热点-95%': '#FF0000',  # 红
        '热点-90%': '#FF6347',  # 番茄红
        '冷点-90%': '#87CEEB',  # 天蓝
        '冷点-95%': '#0000FF',  # 蓝
        '冷点-99%': '#00008B',  # 深蓝
        '不显著': '#D3D3D3'     # 浅灰
    }
    
    for idx, (label, result) in enumerate(results.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[idx]
        
        label_name = result['label_name']
        
        # 获取热点类型列名
        type_col = f'{label_field}_{label}_Type'
        
        if type_col in gdf.columns:
            # 绘制基础地图（灰色）
            gdf.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)
            
            # 绘制热点和冷点
            for pattern_name, color in colors.items():
                # 处理带标签名的类型
                full_pattern = f'{label_name}-{pattern_name}' if pattern_name != '不显著' else '不显著'
                mask = gdf[type_col] == full_pattern
                
                if mask.any():
                    gdf[mask].plot(ax=ax, color=color, edgecolor='white', 
                                  linewidth=0.5, alpha=0.8)
            
            ax.set_title(f'{label_name}\n(n={result["n_features"]}, {result["proportion"]*100:.1f}%)')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'{label_name}\n无数据', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(n_labels, rows * cols):
        axes[idx].axis('off')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color=colors['热点-99%'], label='热点-99%置信'),
        mpatches.Patch(color=colors['热点-95%'], label='热点-95%置信'),
        mpatches.Patch(color=colors['热点-90%'], label='热点-90%置信'),
        mpatches.Patch(color=colors['不显著'], label='不显著'),
        mpatches.Patch(color=colors['冷点-90%'], label='冷点-90%置信'),
        mpatches.Patch(color=colors['冷点-95%'], label='冷点-95%置信'),
        mpatches.Patch(color=colors['冷点-99%'], label='冷点-99%置信')
    ]
    
    fig.legend(handles=legend_elements, loc='center', 
              bbox_to_anchor=(0.5, -0.05), ncol=7)
    
    plt.suptitle(f'分类热点分析: {label_field}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.show()


def create_summary_report(results: Dict, 
                         output_path: Optional[str] = None) -> pd.DataFrame:
    """
    创建热点分析汇总报告
    
    参数:
        results: 热点分析结果
        output_path: 保存路径（可选）
    
    返回:
        汇总DataFrame
    """
    
    summary_data = []
    
    for label, result in results.items():
        # 计算统计
        z_scores = result['Z_score']
        hotspot_types = result['Hotspot_Type']
        
        # 统计各类型数量
        type_counts = hotspot_types.value_counts()
        
        n_hot_99 = sum(1 for t in hotspot_types if '热点-99%' in str(t))
        n_hot_95 = sum(1 for t in hotspot_types if '热点-95%' in str(t))
        n_hot_90 = sum(1 for t in hotspot_types if '热点-90%' in str(t))
        n_cold_90 = sum(1 for t in hotspot_types if '冷点-90%' in str(t))
        n_cold_95 = sum(1 for t in hotspot_types if '冷点-95%' in str(t))
        n_cold_99 = sum(1 for t in hotspot_types if '冷点-99%' in str(t))
        n_not_sig = sum(1 for t in hotspot_types if t == '不显著')
        
        summary_data.append({
            '类别': result['label_name'],
            '要素数': result['n_features'],
            '占比(%)': round(result['proportion'] * 100, 2),
            'Z最小值': round(z_scores.min(), 2),
            'Z最大值': round(z_scores.max(), 2),
            'Z平均值': round(z_scores.mean(), 2),
            '热点-99%': n_hot_99,
            '热点-95%': n_hot_95,
            '热点-90%': n_hot_90,
            '冷点-90%': n_cold_90,
            '冷点-95%': n_cold_95,
            '冷点-99%': n_cold_99,
            '不显著': n_not_sig
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if output_path:
        summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"汇总报告已保存: {output_path}")
    
    return summary_df


def analyze_land_use_hotspots(
    data: Union[str, gpd.GeoDataFrame],
    label_field: str = 'label',
    target_labels: Optional[Union[int, list]] = None,
    generate_report: bool = True,
    generate_plots: bool = True,
    plot_output_dir: Optional[str] = None,
    shp_output_path: Optional[str] = None,
    label_names: Optional[Dict] = None,
    weights: str = 'knn',
    k: int = 8,
    method: str = 'binary',
    verbose: bool = True
) -> tuple:
    """
    分析土地利用热点（重构版）
    
    参数:
        data: 输入数据，可以是：
            - str: shapefile文件路径
            - gpd.GeoDataFrame: 已经读取的GeoDataFrame对象
        label_field: 包含土地利用类型的字段名
        target_labels: 要进行热点分析的标签，可以是：
            - None: 分析所有标签（默认）
            - int: 分析单个标签
            - list: 分析指定的多个标签
        generate_report: 是否生成汇总报告CSV文件
        generate_plots: 是否生成可视化图片
        plot_output_dir: 图片输出目录（如果为None，使用当前目录）
        shp_output_path: shapefile输出路径（如果为None，不输出shapefile）
        label_names: 标签名称映射字典（如果为None，使用默认的LABEL_NAMES）
        weights: 空间权重类型 ('knn' 或 'distance')
        k: k近邻数量
        method: 分析方法 ('binary', 'density', 'count')
        verbose: 是否输出详细信息
    
    返回:
        (gdf, results): 处理后的GeoDataFrame和热点分析结果
    
    示例:
        # 方式1: 从文件路径读取
        gdf, results = analyze_land_use_hotspots(
            data='land_use.shp',
            label_field='label',
            target_labels=[1, 2, 3],
            generate_report=True,
            generate_plots=True,
            plot_output_dir='./outputs',
            shp_output_path='./outputs/hotspots.shp'
        )
        
        # 方式2: 传入已读取的GeoDataFrame
        gdf = gpd.read_file('land_use.shp')
        gdf, results = analyze_land_use_hotspots(
            data=gdf,
            label_field='label',
            target_labels=1,  # 只分析标签1
            generate_plots=False,
            shp_output_path=None  # 不输出shapefile
        )
    """
    
    if verbose:
        print("="*60)
        print("土地利用类型热点分析（重构版）")
        print("="*60)
    
    # 使用默认标签名称如果未提供
    if label_names is None:
        label_names = LABEL_NAMES
    
    # 1. 读取或处理数据
    if verbose:
        print("\n1. 数据准备...")
    
    if isinstance(data, str):
        # 从文件路径读取
        if verbose:
            print(f"   从文件读取: {data}")
        try:
            gdf = gpd.read_file(data, encoding='utf-8')
        except:
            try:
                gdf = gpd.read_file(data, encoding='gbk')
            except:
                gdf = gpd.read_file(data, encoding='gb2312')
        if verbose:
            print(f"   读取 {len(gdf)} 个要素")
    elif isinstance(data, gpd.GeoDataFrame):
        # 使用已有的GeoDataFrame
        gdf = data.copy()
        if verbose:
            print(f"   使用已读取的GeoDataFrame: {len(gdf)} 个要素")
    else:
        raise TypeError("data参数必须是文件路径(str)或GeoDataFrame对象")
    
    if verbose:
        print(f"   字段: {list(gdf.columns)}")
    
    # 2. 数据检查
    if verbose:
        print("\n2. 数据检查...")
    
    # 检查标签字段
    if label_field not in gdf.columns:
        raise ValueError(f"找不到字段 '{label_field}'. 可用字段: {list(gdf.columns)}")
    
    # 统计各类型数量
    label_counts = gdf[label_field].value_counts().sort_index()
    if verbose:
        print("\n   土地利用类型分布:")
        for label, count in label_counts.items():
            name = label_names.get(label, f"未知类型{label}")
            pct = count / len(gdf) * 100
            print(f"   {label}: {name} - {count}个 ({pct:.1f}%)")
    
    # 检查坐标系
    if verbose:
        print(f"\n   坐标系: {gdf.crs}")
        if gdf.crs and gdf.crs.is_geographic:
            print("   警告：使用地理坐标系，将自动转换到投影坐标系")
    
    # 3. 处理目标标签
    if target_labels is not None:
        if isinstance(target_labels, int):
            target_labels = [target_labels]
        
        if verbose:
            print(f"\n   指定分析的标签: {target_labels}")
        
        # 过滤数据或在分析时只处理指定标签
        # 这里我们创建一个临时的label_names字典
        filtered_label_names = {k: v for k, v in label_names.items() if k in target_labels}
    else:
        filtered_label_names = label_names
    
    # 4. 执行热点分析
    if verbose:
        print("\n3. 执行热点分析...")
        print("-"*40)
    
    # 如果指定了target_labels，临时修改gdf
    if target_labels is not None:
        # 创建临时副本，将非目标标签设为NaN
        gdf_temp = gdf.copy()
        mask = ~gdf_temp[label_field].isin(target_labels)
        gdf_temp.loc[mask, label_field] = np.nan
        
        results = compute_categorical_hotspot(
            gdf_temp,
            label_field=label_field,
            label_names=filtered_label_names,
            weights=weights,
            k=k,
            method=method,
            auto_fix_projection=True,
            add_to_gdf=True,  # 结果会添加到gdf_temp
            verbose=verbose
        )
        
        # 将结果字段复制到原始gdf
        for col in gdf_temp.columns:
            if col not in gdf.columns and ('_Z' in col or '_P' in col or '_Type' in col):
                gdf[col] = gdf_temp[col]
    else:
        results = compute_categorical_hotspot(
            gdf,
            label_field=label_field,
            label_names=filtered_label_names,
            weights=weights,
            k=k,
            method=method,
            auto_fix_projection=True,
            add_to_gdf=True,
            verbose=verbose
        )
    
    # 5. 生成汇总报告
    if generate_report:
        if verbose:
            print("\n4. 生成汇总报告...")
        
        if plot_output_dir:
            Path(plot_output_dir).mkdir(parents=True, exist_ok=True)
            report_path = Path(plot_output_dir) / 'land_use_hotspot_summary.csv'
        else:
            report_path = 'land_use_hotspot_summary.csv'
        
        summary = create_summary_report(results, output_path=str(report_path))
        
        if verbose:
            print("\n" + "="*60)
            print("热点分析汇总")
            print("="*60)
            print(summary.to_string())
    
    # 6. 创建可视化
    if generate_plots:
        if verbose:
            print("\n5. 创建可视化...")
        
        # 创建输出目录
        if plot_output_dir:
            Path(plot_output_dir).mkdir(parents=True, exist_ok=True)
            dist_plot_path = Path(plot_output_dir) / 'land_use_distribution.png'
            hotspot_plot_path = Path(plot_output_dir) / 'land_use_hotspots.png'
        else:
            dist_plot_path = 'land_use_distribution.png'
            hotspot_plot_path = 'land_use_hotspots.png'
        
        # 整体分布图
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 定义颜色
        land_use_colors = {
            1: '#FF6B6B',  # 商业 - 红色
            2: '#4ECDC4',  # 住宅 - 青色
            3: '#45B7D1',  # 公共服务 - 蓝色
            4: '#96CEB4',  # 科技与工业 - 绿色
            5: '#FFEAA7'   # 教育文化 - 黄色
        }
        
        # 绘制原始分布
        labels_to_plot = target_labels if target_labels else land_use_colors.keys()
        for label in labels_to_plot:
            if label in land_use_colors:
                color = land_use_colors[label]
                mask = gdf[label_field] == label
                if mask.any():
                    name = label_names.get(label, str(label))
                    gdf[mask].plot(ax=ax, color=color, label=name, alpha=0.7)
        
        ax.set_title('土地利用类型分布', fontsize=14)
        ax.legend(loc='upper right')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(dist_plot_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"   分布图已保存: {dist_plot_path}")
        plt.close()
        
        # 热点分析可视化
        visualize_categorical_hotspots(
            gdf,
            label_field,
            results,
            filtered_label_names,
            figsize=(15, 10),
            save_path=str(hotspot_plot_path)
        )
    
    # 7. 导出结果
    if shp_output_path:
        if verbose:
            print("\n6. 导出shapefile结果...")
        
        # 创建输出目录
        output_dir = Path(shp_output_path).parent
        if output_dir != Path('.'):
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 选择要导出的字段
        export_columns = ['geometry', label_field]
        
        # 添加所有热点分析结果字段
        for col in gdf.columns:
            if '_Z' in col or '_P' in col or '_Type' in col:
                export_columns.append(col)
        
        # 确保所有字段都存在
        export_columns = [col for col in export_columns if col in gdf.columns]
        
        # 导出为新的shapefile
        gdf[export_columns].to_file(shp_output_path, encoding='utf-8')
        if verbose:
            print(f"   结果shapefile已保存: {shp_output_path}")
        
        # 导出为CSV（不含几何）
        csv_path = str(Path(shp_output_path).with_suffix('.csv'))
        gdf[export_columns].drop('geometry', axis=1).to_csv(
            csv_path, index=False, encoding='utf-8-sig'
        )
        if verbose:
            print(f"   结果CSV已保存: {csv_path}")
    
    if verbose:
        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)
    
    return gdf, results


if __name__ == "__main__":
    # 模块测试示例
    print("Shapefile工具模块")
    print("="*60)
    print("可用函数列表:")
    print("\n【文件操作】")
    print("- read_shapefile, write_shapefile, convert_format")
    print("- read_multiple_shapefiles")
    
    print("\n【投影处理】")
    print("- get_crs_info, transform_crs, copy_crs_from_file")
    
    print("\n【统计分析】")
    print("- get_basic_stats, get_geometry_stats, get_attribute_stats")
    print("- get_field_summary, group_statistics, count_by_field")
    
    print("\n【属性字段操作】⭐新增")
    print("- get_field_names: 查询所有字段名")
    print("- add_field: 添加新字段")
    print("- delete_field, delete_fields: 删除字段")
    print("- rename_field: 重命名字段")
    print("- calculate_field: 计算字段值")
    print("- get_unique_values: 获取唯一值")
    
    print("\n【数据筛选】⭐增强")
    print("- filter_by_attribute: 按条件筛选")
    print("- filter_by_values: 按值列表筛选")
    print("- filter_by_range: 按数值范围筛选")
    print("- filter_by_geometry: 空间筛选")
    
    print("\n【数据处理】")
    print("- add_area_length_fields, buffer_geometries")
    print("- dissolve_by_field, clip_by_boundary")
    print("- calculate_centroids")
    print("- sample_features, merge_by_field")
    
    print("\n【中心化与标准化】⭐新增（用于莫兰指数等）")
    print("- centralize_coordinates: 坐标中心化")
    print("- add_centroid_coordinates: 添加中心点坐标")
    print("- standardize_values: 标准化字段值")
    print("- compute_morans_i: 计算全局莫兰指数")
    
    print("\n【空间分析】")
    print("- spatial_join, calculate_distance_matrix")
    print("- find_nearest_features")
    
    print("\n【数据验证】")
    print("- check_geometry_validity, fix_invalid_geometries")
    print("- remove_duplicate_geometries")
    
    print("\n【实用工具】")
    print("- print_stats_summary, export_to_csv, get_file_info")
    
    print("\n【功能性分析】⭐新增")
    print("- shannon_entropy, simpson_index: 多样性指标")
    print("- functional_diversity_analysis: 功能多样性分析")
    print("- compute_morans_i_by_label: 各标签莫兰指数")
    print("- compute_getis_ord_gi_star: 热点分析（Gi*）")
    print("- hotspot_analysis_by_label: 批量热点分析")
    
    print("\n" + "="*60)
    print("使用示例:")
    print("from shp_utils import *")
    print("\n# 查看字段")
    print("fields = get_field_names(gdf)")
    print("\n# 添加字段")
    print("gdf = add_field(gdf, 'new_field', 0)")
    print("\n# 筛选数据")
    print("filtered = filter_by_values(gdf, 'category', ['A', 'B'])")
    print("\n# 中心化（用于莫兰指数）")
    print("gdf = centralize_coordinates(gdf, add_fields=True)")