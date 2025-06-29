import sys
import os
import shutil
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                           QFileDialog, QTextEdit, QSplitter, QGroupBox,
                           QGridLayout, QMessageBox, QProgressBar, QComboBox,
                           QCheckBox, QSpinBox, QDoubleSpinBox, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QPen, QBrush
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import requests
from io import BytesIO
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 在线地图瓦片URL配置
MAP_TILES = {
    'OpenStreetMap': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    'CartoDB Dark': 'https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png',
    'CartoDB Positron': 'https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png',
    'Stamen Terrain': 'https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg',
}

class MapCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 8), facecolor='#1e1e1e')
        super().__init__(self.fig)
        self.setParent(parent)
        
        # 设置深色主题
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        self.setup_map()
        
        # 鼠标事件
        self.press = None
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('scroll_event', self.on_scroll)
        
    def setup_map(self):
        """设置地图样式，并隐藏坐标轴"""
        self.ax.set_facecolor('#0a0a0a')
        self.ax.grid(False)  # 关闭网格
        self.ax.axis('off')  # 隐藏坐标轴
        
        
    def change_map_style(self, style):
        """更改地图样式"""
        self.map_style = style
        self.refresh_map()
        
    def refresh_map(self):
        """刷新地图底图"""
        try:
            # 获取当前视图范围
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # 添加在线地图瓦片
            if self.map_style in MAP_TILES:
                ctx.add_basemap(self.ax, crs='EPSG:4326', 
                              source=MAP_TILES[self.map_style],
                              alpha=0.8, attribution='')
            
            # 恢复视图范围
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            
        except Exception as e:
            print(f"无法加载在线地图: {e}")
            # 如果在线地图加载失败，使用默认样式
            pass
        
    def plot_trajectory(self, df, color='#00aaff', alpha=0.9, linewidth=2.5):
        """绘制单条轨迹"""
        if len(df) > 0:
            lons = df['lon'].values
            lats = df['lat'].values
            
            # 绘制轨迹线
            self.ax.plot(lons, lats, color=color, alpha=alpha, linewidth=linewidth, zorder=5)
            
            # 绘制起点和终点
            self.ax.scatter(lons[0], lats[0], color='#ff6b6b', s=120, marker='o', 
                          label='Start', zorder=6, edgecolors='white', linewidth=1)
            self.ax.scatter(lons[-1], lats[-1], color='#4ecdc4', s=120, marker='s', 
                          label='End', zorder=6, edgecolors='white', linewidth=1)
            
            # 自动调整视图范围
            margin = 0.01
            self.ax.set_xlim(min(lons) - margin, max(lons) + margin)
            self.ax.set_ylim(min(lats) - margin, max(lats) + margin)
            
            # 刷新地图底图
            self.refresh_map()
            self.draw()
    
    def clear_trajectories(self):
        """清除所有轨迹"""
        self.ax.clear()
        self.setup_map()
        self.draw()
    
    def plot_selection_area(self, min_lat, max_lat, min_lon, max_lon):
        """绘制选择区域"""
        rect = Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                        linewidth=3, edgecolor='#ffd93d', facecolor='#ffd93d', alpha=0.2, zorder=4)
        self.ax.add_patch(rect)
        self.draw()
    
    def on_scroll(self, event):
        """鼠标滚轮缩放事件"""
        if event.inaxes != self.ax:
            return
        
        # 缩放因子
        scale_factor = 1.1
        if event.button == 'up':
            scale_factor = 1 / scale_factor
        
        # 获取当前范围
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # 计算新的范围
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        
        new_xrange = xrange * scale_factor
        new_yrange = yrange * scale_factor
        
        # 以鼠标位置为中心缩放
        center_x = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
        center_y = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2
        
        new_xlim = [center_x - new_xrange/2, center_x + new_xrange/2]
        new_ylim = [center_y - new_yrange/2, center_y + new_yrange/2]
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.draw()
    
    def on_press(self, event):
        """鼠标按下事件"""
        if event.inaxes != self.ax:
            return
        self.press = (event.xdata, event.ydata)
    
    def on_motion(self, event):
        """鼠标移动事件"""
        if self.press is None:
            return
        if event.inaxes != self.ax:
            return
        
        # 实现地图拖拽
        dx = event.xdata - self.press[0]
        dy = event.ydata - self.press[1]
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        self.draw()
    
    def on_release(self, event):
        """鼠标释放事件"""
        self.press = None
        self.draw()

class TrajectoryProcessor(QThread):
    progress_updated = pyqtSignal(int)
    file_processed = pyqtSignal(str, bool)
    finished_processing = pyqtSignal(list)
    
    def __init__(self, folder_path, min_lat, max_lat, min_lon, max_lon):
        super().__init__()
        self.folder_path = folder_path
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        
    def run(self):
        """处理轨迹文件，筛选经过指定区域的轨迹"""
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        filtered_files = []
        
        for i, filename in enumerate(csv_files):
            filepath = os.path.join(self.folder_path, filename)
            try:
                df = pd.read_csv(filepath)
                
                # 检查是否包含必要的列
                required_cols = ['lat', 'lon']
                if not all(col in df.columns for col in required_cols):
                    self.file_processed.emit(filename, False)
                    continue
                
                # 检查轨迹是否经过指定区域
                in_area = ((df['lat'] >= self.min_lat) & (df['lat'] <= self.max_lat) & 
                          (df['lon'] >= self.min_lon) & (df['lon'] <= self.max_lon))
                
                if in_area.any():
                    filtered_files.append(filepath)
                    self.file_processed.emit(filename, True)
                else:
                    self.file_processed.emit(filename, False)
                    
            except Exception as e:
                self.file_processed.emit(filename, False)
            
            # 更新进度
            progress = int((i + 1) / len(csv_files) * 100)
            self.progress_updated.emit(progress)
        
        self.finished_processing.emit(filtered_files)

class ShipTrajectorySystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("海洋船舶轨迹处理可视化系统")
        self.setGeometry(100, 100, 1400, 800)
        
        # 设置深色主题
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QPushButton {
                background-color: #2d2d2d;
                border: 2px solid #007acc;
                color: #ffffff;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border-color: #00aaff;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #1d1d1d;
                border-color: #0088cc;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                border-color: #333333;
                color: #666666;
            }
            QLineEdit, QDoubleSpinBox, QComboBox {
                background-color: #2d2d2d;
                border: 2px solid #444444;
                color: #ffffff;
                padding: 6px;
                border-radius: 4px;
                font-size: 11px;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #007acc;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #3d3d3d;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                border: 1px solid #007acc;
                selection-background-color: #007acc;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #007acc;
                border-radius: 6px;
                margin-top: 1ex;
                color: #ffffff;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #00aaff;
                font-weight: bold;
            }
            QTextEdit {
                background-color: #2d2d2d;
                border: 2px solid #444444;
                color: #ffffff;
                border-radius: 4px;
                font-size: 11px;
            }
            QTabWidget::pane {
                border: 2px solid #007acc;
                border-radius: 4px;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #1e1e1e;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border: 2px solid #444444;
            }
            QTabBar::tab:selected {
                background-color: #007acc;
                border-color: #007acc;
            }
            QTabBar::tab:hover {
                background-color: #3d3d3d;
            }
            QProgressBar {
                border: 2px solid #444444;
                border-radius: 5px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 3px;
            }
        """)
        
        self.setup_ui()
        self.current_trajectory_files = []
        self.current_file_index = 0
        self.save_folder = ""
        
    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧工具面板
        tool_panel = self.create_tool_panel()
        splitter.addWidget(tool_panel)
        
        # 右侧地图面板
        map_panel = self.create_map_panel()
        splitter.addWidget(map_panel)
        
        # 设置分割器比例
        splitter.setSizes([400, 1000])
    
    def create_tool_panel(self):
        """创建工具面板"""
        tool_widget = QWidget()
        tool_layout = QVBoxLayout(tool_widget)
        
        # 文件选择组
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout(file_group)
        
        self.select_folder_btn = QPushButton("选择文件夹")
        self.select_folder_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(self.select_folder_btn)
        
        self.select_file_btn = QPushButton("选择单个文件")
        self.select_file_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.select_file_btn)
        
        self.file_path_label = QLabel("未选择文件")
        file_layout.addWidget(self.file_path_label)
        
        tool_layout.addWidget(file_group)
        
        # 地图样式选择组
        map_style_group = QGroupBox("地图样式")
        map_style_layout = QVBoxLayout(map_style_group)
        
        self.map_style_combo = QComboBox()
        self.map_style_combo.addItems(['CartoDB Dark', 'OpenStreetMap', 'CartoDB Positron', 'Stamen Terrain'])
        self.map_style_combo.currentTextChanged.connect(self.change_map_style)
        map_style_layout.addWidget(self.map_style_combo)
        
        tool_layout.addWidget(map_style_group)
        
        # 区域筛选组
        filter_group = QGroupBox("区域筛选")
        filter_layout = QGridLayout(filter_group)
        
        filter_layout.addWidget(QLabel("最小纬度:"), 0, 0)
        self.min_lat_input = QDoubleSpinBox()
        self.min_lat_input.setRange(-90, 90)
        self.min_lat_input.setValue(0)
        self.min_lat_input.setDecimals(6)
        filter_layout.addWidget(self.min_lat_input, 0, 1)
        
        filter_layout.addWidget(QLabel("最大纬度:"), 1, 0)
        self.max_lat_input = QDoubleSpinBox()
        self.max_lat_input.setRange(-90, 90)
        self.max_lat_input.setValue(10)
        self.max_lat_input.setDecimals(6)
        filter_layout.addWidget(self.max_lat_input, 1, 1)
        
        filter_layout.addWidget(QLabel("最小经度:"), 2, 0)
        self.min_lon_input = QDoubleSpinBox()
        self.min_lon_input.setRange(-180, 180)
        self.min_lon_input.setValue(100)
        self.min_lon_input.setDecimals(6)
        filter_layout.addWidget(self.min_lon_input, 2, 1)
        
        filter_layout.addWidget(QLabel("最大经度:"), 3, 0)
        self.max_lon_input = QDoubleSpinBox()
        self.max_lon_input.setRange(-180, 180)
        self.max_lon_input.setValue(120)
        self.max_lon_input.setDecimals(6)
        filter_layout.addWidget(self.max_lon_input, 3, 1)
        
        self.filter_btn = QPushButton("筛选轨迹")
        self.filter_btn.clicked.connect(self.filter_trajectories)
        filter_layout.addWidget(self.filter_btn, 4, 0, 1, 2)
        
        self.show_area_btn = QPushButton("显示筛选区域")
        self.show_area_btn.clicked.connect(self.show_selection_area)
        filter_layout.addWidget(self.show_area_btn, 5, 0, 1, 2)
        
        tool_layout.addWidget(filter_group)
        
        # 轨迹控制组
        control_group = QGroupBox("轨迹控制")
        control_layout = QVBoxLayout(control_group)
        
        self.trajectory_info_label = QLabel("当前轨迹: 0/0")
        control_layout.addWidget(self.trajectory_info_label)
        
        self.next_btn = QPushButton("下一条轨迹")
        self.next_btn.clicked.connect(self.next_trajectory)
        self.next_btn.setEnabled(False)
        control_layout.addWidget(self.next_btn)
        
        self.prev_btn = QPushButton("上一条轨迹")
        self.prev_btn.clicked.connect(self.prev_trajectory)
        self.prev_btn.setEnabled(False)
        control_layout.addWidget(self.prev_btn)
        
        self.save_btn = QPushButton("保存当前轨迹")
        self.save_btn.clicked.connect(self.save_current_trajectory)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)
        
        self.select_save_folder_btn = QPushButton("选择保存文件夹")
        self.select_save_folder_btn.clicked.connect(self.select_save_folder)
        control_layout.addWidget(self.select_save_folder_btn)
        
        self.save_folder_label = QLabel("未选择保存文件夹")
        control_layout.addWidget(self.save_folder_label)
        
        tool_layout.addWidget(control_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        tool_layout.addWidget(self.progress_bar)
        
        # 创建选项卡容器
        tab_widget = QTabWidget()
        
        # 日志选项卡
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        tab_widget.addTab(log_tab, "操作日志")
        
        # 统计信息选项卡
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        tab_widget.addTab(stats_tab, "文件统计")
        
        tool_layout.addWidget(tab_widget)
        
        # 清除按钮
        self.clear_btn = QPushButton("清除地图")
        self.clear_btn.clicked.connect(self.clear_map)
        tool_layout.addWidget(self.clear_btn)
        
        tool_layout.addStretch()
        
        return tool_widget
    
    def create_map_panel(self):
        """创建地图面板"""
        map_widget = QWidget()
        map_layout = QVBoxLayout(map_widget)
        
        # 地图标题
        title_label = QLabel("海洋船舶轨迹态势图")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00aaff; padding: 10px;")
        map_layout.addWidget(title_label)
        
        # 地图画布
        self.map_canvas = MapCanvas()
        map_layout.addWidget(self.map_canvas)
        
        return map_widget
    
    def select_folder(self):
        """选择文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择轨迹文件夹")
        if folder:
            self.file_path_label.setText(f"文件夹: {folder}")
            self.current_folder = folder
            self.log_message(f"选择文件夹: {folder}")
            self.analyze_folder_statistics(folder)
    
    def analyze_folder_statistics(self, folder_path):
        """分析文件夹统计信息"""
        try:
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            total_files = len(csv_files)
            
            if total_files == 0:
                self.update_statistics("未找到CSV文件")
                return
            
            valid_files = 0
            total_points = 0
            date_range = {'min': None, 'max': None}
            lat_range = {'min': float('inf'), 'max': float('-inf')}
            lon_range = {'min': float('inf'), 'max': float('-inf')}
            file_sizes = []
            
            self.log_message(f"开始分析 {total_files} 个CSV文件...")
            
            for i, filename in enumerate(csv_files):
                try:
                    filepath = os.path.join(folder_path, filename)
                    df = pd.read_csv(filepath)
                    
                    # 检查必要列
                    if 'lat' in df.columns and 'lon' in df.columns:
                        valid_files += 1
                        points_count = len(df)
                        total_points += points_count
                        file_sizes.append(points_count)
                        
                        # 更新经纬度范围
                        lat_range['min'] = min(lat_range['min'], df['lat'].min())
                        lat_range['max'] = max(lat_range['max'], df['lat'].max())
                        lon_range['min'] = min(lon_range['min'], df['lon'].min())
                        lon_range['max'] = max(lon_range['max'], df['lon'].max())
                        
                        # 处理日期范围
                        if 'date' in df.columns:
                            try:
                                df['date'] = pd.to_datetime(df['date'])
                                file_min_date = df['date'].min()
                                file_max_date = df['date'].max()
                                
                                if date_range['min'] is None or file_min_date < date_range['min']:
                                    date_range['min'] = file_min_date
                                if date_range['max'] is None or file_max_date > date_range['max']:
                                    date_range['max'] = file_max_date
                            except:
                                pass
                    
                    # 更新进度
                    if i % 10 == 0:  # 每10个文件更新一次进度
                        progress = int((i + 1) / total_files * 100)
                        self.progress_bar.setValue(progress)
                
                except Exception as e:
                    continue
            
            # 生成统计报告
            stats_report = f"""📊 文件夹统计信息
{'='*40}
📁 总文件数: {total_files}
✅ 有效轨迹文件: {valid_files}
❌ 无效文件: {total_files - valid_files}

📈 轨迹数据统计
{'='*40}
🎯 总轨迹点数: {total_points:,}
📏 平均每文件点数: {int(total_points/valid_files) if valid_files > 0 else 0}
📊 最大文件点数: {max(file_sizes) if file_sizes else 0}
📉 最小文件点数: {min(file_sizes) if file_sizes else 0}

🗺️ 地理范围
{'='*40}
🌍 纬度范围: {lat_range['min']:.6f} ~ {lat_range['max']:.6f}
🌐 经度范围: {lon_range['min']:.6f} ~ {lon_range['max']:.6f}
📐 纬度跨度: {lat_range['max'] - lat_range['min']:.6f}°
📐 经度跨度: {lon_range['max'] - lon_range['min']:.6f}°

⏰ 时间范围
{'='*40}"""
            
            if date_range['min'] and date_range['max']:
                stats_report += f"""
📅 开始时间: {date_range['min'].strftime('%Y-%m-%d %H:%M:%S')}
📅 结束时间: {date_range['max'].strftime('%Y-%m-%d %H:%M:%S')}
⏱️ 时间跨度: {(date_range['max'] - date_range['min']).days} 天"""
            else:
                stats_report += f"""
📅 时间信息: 无法解析日期字段"""
            
            self.update_statistics(stats_report)
            self.progress_bar.setValue(0)
            
        except Exception as e:
            self.update_statistics(f"分析失败: {str(e)}")
            self.log_message(f"统计分析出错: {str(e)}")
    
    def update_statistics(self, stats_text):
        """更新统计信息显示"""
        self.stats_text.setPlainText(stats_text)
    
    def change_map_style(self, style):
        """更改地图样式"""
        self.map_canvas.change_map_style(style)
        # self.log_message(f"切换地图样式: {style}")选择文件夹: {folder}")
        # self.log_message(f"切换地图样式: {style}, 选择文件夹: {folder}")
        self.log_message(f"切换地图样式: {style}")
    
    def select_file(self):
        """选择单个文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择轨迹文件", "", "CSV Files (*.csv)")
        if file_path:
            self.file_path_label.setText(f"文件: {os.path.basename(file_path)}")
            self.load_single_file(file_path)
    
    def load_single_file(self, file_path):
        """加载单个文件"""
        try:
            df = pd.read_csv(file_path)
            self.map_canvas.clear_trajectories()
            self.map_canvas.plot_trajectory(df)
            self.log_message(f"加载文件: {os.path.basename(file_path)}")
        except Exception as e:
            self.log_message(f"加载文件失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"加载文件失败: {str(e)}")
    
    def filter_trajectories(self):
        """筛选轨迹"""
        if not hasattr(self, 'current_folder'):
            QMessageBox.warning(self, "警告", "请先选择文件夹")
            return
        
        min_lat = self.min_lat_input.value()
        max_lat = self.max_lat_input.value()
        min_lon = self.min_lon_input.value()
        max_lon = self.max_lon_input.value()
        
        if min_lat >= max_lat or min_lon >= max_lon:
            QMessageBox.warning(self, "警告", "经纬度范围设置错误")
            return
        
        self.log_message(f"开始筛选轨迹，区域: ({min_lat}, {min_lon}) - ({max_lat}, {max_lon})")
        
        # 创建处理线程
        self.processor = TrajectoryProcessor(self.current_folder, min_lat, max_lat, min_lon, max_lon)
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.file_processed.connect(self.on_file_processed)
        self.processor.finished_processing.connect(self.on_filtering_finished)
        self.processor.start()
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def on_file_processed(self, filename, in_area):
        """处理单个文件完成"""
        if in_area:
            self.log_message(f"✓ {filename} 经过目标区域")
        else:
            self.log_message(f"✗ {filename} 未经过目标区域")
    
    def on_filtering_finished(self, filtered_files):
        """筛选完成"""
        self.current_trajectory_files = filtered_files
        self.current_file_index = 0
        
        if filtered_files:
            self.trajectory_info_label.setText(f"当前轨迹: 1/{len(filtered_files)}")
            self.next_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.show_current_trajectory()
            self.log_message(f"筛选完成，找到 {len(filtered_files)} 条符合条件的轨迹")
        else:
            self.trajectory_info_label.setText("当前轨迹: 0/0")
            self.next_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.log_message("未找到符合条件的轨迹")
            QMessageBox.information(self, "提示", "未找到符合条件的轨迹")
        
        self.progress_bar.setValue(0)
    
    def show_current_trajectory(self):
        """显示当前轨迹"""
        if not self.current_trajectory_files:
            return
        
        current_file = self.current_trajectory_files[self.current_file_index]
        try:
            df = pd.read_csv(current_file)
            self.map_canvas.clear_trajectories()
            self.map_canvas.plot_trajectory(df)
            
            # 更新信息
            filename = os.path.basename(current_file)
            self.trajectory_info_label.setText(
                f"当前轨迹: {self.current_file_index + 1}/{len(self.current_trajectory_files)}"
            )
            self.log_message(f"显示轨迹: {filename}")
            
        except Exception as e:
            self.log_message(f"显示轨迹失败: {str(e)}")
    
    def next_trajectory(self):
        """下一条轨迹"""
        if self.current_file_index < len(self.current_trajectory_files) - 1:
            self.current_file_index += 1
            self.show_current_trajectory()
    
    def prev_trajectory(self):
        """上一条轨迹"""
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.show_current_trajectory()
    
    def save_current_trajectory(self):
        """保存当前轨迹"""
        if not self.save_folder:
            QMessageBox.warning(self, "警告", "请先选择保存文件夹")
            return
        
        if not self.current_trajectory_files:
            return
        
        current_file = self.current_trajectory_files[self.current_file_index]
        filename = os.path.basename(current_file)
        destination = os.path.join(self.save_folder, filename)
        
        try:
            shutil.copy2(current_file, destination)
            self.log_message(f"保存成功: {filename}")
            QMessageBox.information(self, "成功", f"轨迹文件已保存到: {destination}")
        except Exception as e:
            self.log_message(f"保存失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")
    
    def select_save_folder(self):
        """选择保存文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if folder:
            self.save_folder = folder
            self.save_folder_label.setText(f"保存到: {folder}")
            self.log_message(f"选择保存文件夹: {folder}")
    
    def show_selection_area(self):
        """显示筛选区域"""
        min_lat = self.min_lat_input.value()
        max_lat = self.max_lat_input.value()
        min_lon = self.min_lon_input.value()
        max_lon = self.max_lon_input.value()
        
        self.map_canvas.plot_selection_area(min_lat, max_lat, min_lon, max_lon)
        self.log_message("显示筛选区域")
    
    def clear_map(self):
        """清除地图"""
        self.map_canvas.clear_trajectories()
        self.log_message("清除地图")
    
    def log_message(self, message):
        """记录日志消息"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.ensureCursorVisible()

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序图标和名称
    app.setApplicationName("海洋船舶轨迹处理可视化系统")
    app.setApplicationVersion("1.0")
    
    window = ShipTrajectorySystem()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()