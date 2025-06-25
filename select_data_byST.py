import streamlit as st
import pandas as pd
import os
import shutil
import folium
from streamlit_folium import st_folium
from pathlib import Path
import time

# 初始化session_state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'filtered_files' not in st.session_state:
    st.session_state.filtered_files = []
if 'map' not in st.session_state:
    st.session_state.map = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {
        'time': 'date',
        'longitude': 'lon',
        'latitude': 'lat',
        'speed': 'sog',
        'heading': 'cog'
    }

# 创建目标文件夹
def create_save_folder():
    save_path = Path("./saved_tracks")
    save_path.mkdir(exist_ok=True)
    return save_path

# 绘制单条航迹
def plot_track_on_map(df, map_obj, column_mapping):
    # 获取映射后的列名
    lat_col = column_mapping['latitude']
    lon_col = column_mapping['longitude']
    
    # 绘制完整航迹
    track_points = list(zip(df[lat_col], df[lon_col]))
    folium.PolyLine(track_points, color='blue', weight=2.5, opacity=1).add_to(map_obj)
    
    # 标记起点和终点
    folium.Marker(
        location=track_points[0],
        icon=folium.Icon(color='green', icon='play', prefix='fa'),
        tooltip="起点"
    ).add_to(map_obj)
    
    folium.Marker(
        location=track_points[-1],
        icon=folium.Icon(color='red', icon='stop', prefix='fa'),
        tooltip="终点"
    ).add_to(map_obj)
    
    # 设置地图范围
    sw = df[[lat_col, lon_col]].min().values.tolist()
    ne = df[[lat_col, lon_col]].max().values.tolist()
    map_obj.fit_bounds([sw, ne])
    
    return map_obj

# 主应用
def main():
    st.title("🚢 AIS航迹数据筛选工具")
    st.write("选择包含CSV航迹数据的文件夹，设置筛选区域，然后交互式保存航迹")
    
    # 创建保存文件夹
    save_folder = create_save_folder()
    
    # 文件夹选择
    data_dir = st.text_input("航迹数据文件夹路径", "./sample_data")
    
    # 列名映射设置
    st.subheader("列名映射设置")
    st.info("如果您的CSV文件使用不同的列名，请在此指定映射关系")
    
    with st.expander("配置列名映射"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.column_mapping['time'] = st.text_input("时间列名", 
                                                                    st.session_state.column_mapping['time'],
                                                                    help="如: timestamp, 时间, datetime")
            st.session_state.column_mapping['longitude'] = st.text_input("经度列名", 
                                                                       st.session_state.column_mapping['longitude'],
                                                                       help="如: lon, lng, 经度, longitude")
            st.session_state.column_mapping['latitude'] = st.text_input("纬度列名", 
                                                                      st.session_state.column_mapping['latitude'],
                                                                      help="如: lat, 纬度, latitude")
        with col2:
            st.session_state.column_mapping['speed'] = st.text_input("速度列名", 
                                                                    st.session_state.column_mapping['speed'],
                                                                    help="如: sog, 速度, velocity")
            st.session_state.column_mapping['heading'] = st.text_input("航向列名", 
                                                                     st.session_state.column_mapping['heading'],
                                                                     help="如: cog, 航向, course")
    
    # 区域选择
    st.subheader("区域筛选条件")
    col1, col2 = st.columns(2)
    with col1:
        min_lat = st.number_input("最小纬度", value=20.0)
        max_lat = st.number_input("最大纬度", value=32.0)
    with col2:
        min_lon = st.number_input("最小经度", value=110.0)
        max_lon = st.number_input("最大经度", value=130.0)
    
    # 筛选按钮
    if st.button("筛选航迹数据"):
        all_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".csv"):
                    all_files.append(os.path.join(root, file))
        
        if not all_files:
            st.warning("未找到CSV文件！")
            return
        
        filtered = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 获取映射后的列名
        lat_col = st.session_state.column_mapping['latitude']
        lon_col = st.session_state.column_mapping['longitude']
        
        for i, file_path in enumerate(all_files):
            try:
                df = pd.read_csv(file_path)

                # 检查列名是否存在于数据中
                required_cols = {lat_col, lon_col}
                if not required_cols.issubset(df.columns):
                    missing_cols = required_cols - set(df.columns)
                    st.warning(f"文件 {os.path.basename(file_path)} 缺少必要列: {', '.join(missing_cols)}")
                    continue
                
                # 筛选区域内的点
                in_area = df[
                    (df[lat_col].between(min_lat, max_lat)) & 
                    (df[lon_col].between(min_lon, max_lon))
                ]
                if not in_area.empty:
                    filtered.append((file_path, len(df)))
                
            except Exception as e:
                st.error(f"处理文件 {file_path} 时出错: {str(e)}")
            
            progress_bar.progress((i + 1) / len(all_files))
            status_text.text(f"处理中: {i+1}/{len(all_files)} 文件")
        
        # 排序结果
        st.session_state.filtered_files = sorted(filtered, key=lambda x: x[0])
        st.session_state.current_index = 0
        st.success(f"找到 {len(filtered)} 条经过该区域的航迹!")
    
    # 显示筛选结果
    if st.session_state.filtered_files:
        st.subheader(f"筛选结果: {len(st.session_state.filtered_files)} 条航迹")
        current_file, point_count = st.session_state.filtered_files[st.session_state.current_index]
        st.write(f"当前航迹: {os.path.basename(current_file)} (点数: {point_count})")
        
        # 加载当前航迹数据
        try:
            current_df = pd.read_csv(current_file)
            
            # 创建地图
            if st.session_state.map is None:
                center_lat = (min_lat + max_lat) / 2
                center_lon = (min_lon + max_lon) / 2
                st.session_state.map = folium.Map(location=[center_lat, center_lon], zoom_start=9)
                
                # 绘制筛选区域
                area_coords = [
                    [min_lat, min_lon],
                    [min_lat, max_lon],
                    [max_lat, max_lon],
                    [max_lat, min_lon]
                ]
                folium.Rectangle(
                    bounds=area_coords,
                    color='#ff7800',
                    fill=True,
                    fill_color='#ffff00',
                    fill_opacity=0.2,
                    weight=2,
                    tooltip="筛选区域"
                ).add_to(st.session_state.map)
            
            # 绘制当前航迹
            st.session_state.map = plot_track_on_map(current_df, 
                                                   st.session_state.map, 
                                                   st.session_state.column_mapping)
            
        except Exception as e:
            st.error(f"加载航迹数据时出错: {str(e)}")
        
        # 显示地图
        st_map = st_folium(st.session_state.map, width=800, height=500)
        
        # 航迹控制按钮
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ 上一条", disabled=st.session_state.current_index==0):
                st.session_state.current_index -= 1
                st.experimental_rerun()
        
        with col2:
            save_btn = st.button("💾 保存当前航迹")
            if save_btn:
                try:
                    dest_path = save_folder / os.path.basename(current_file)
                    shutil.copy(current_file, dest_path)
                    st.success(f"已保存: {dest_path}")
                    time.sleep(1)
                except Exception as e:
                    st.error(f"保存失败: {str(e)}")
        
        with col3:
            if st.button("➡️ 下一条", disabled=st.session_state.current_index>=len(st.session_state.filtered_files)-1):
                st.session_state.current_index += 1
                st.experimental_rerun()
        
        # 显示进度
        st.progress((st.session_state.current_index + 1) / len(st.session_state.filtered_files))
        st.write(f"航迹 {st.session_state.current_index + 1}/{len(st.session_state.filtered_files)}")
        
        # 显示数据预览
        with st.expander("查看当前航迹数据"):
            st.dataframe(current_df.head(10))

if __name__ == "__main__":
    main()