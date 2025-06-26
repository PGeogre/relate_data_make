import os
import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import QWebEngineView
import folium
from folium.plugins import MarkerCluster
from io import BytesIO
from PIL import Image, ImageEnhance
import geopandas as gpd
from shapely.geometry import Point, Polygon

class ShipTrackVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("船舶轨迹态势可视化系统")
        self.setGeometry(100, 100, 1600, 900)
        self.setup_ui()
        self.ship_data = {}
        self.current_map = None
        self.map_html = ""
        self.dark_theme = True
        self.apply_dark_theme()

    def setup_ui(self):
        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        control_panel = QFrame()
        control_panel.setFixedWidth(350)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)
        control_layout.setSpacing(20)

        # 标题
        title = QLabel("船舶轨迹态势分析")
        title.setStyleSheet("font-size: 20pt; font-weight: bold; color: #1E90FF;")
        title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title)

        # 分隔线
        control_layout.addWidget(self.create_h_line())

        # 文件夹选择
        folder_group = QGroupBox("数据源")
        folder_layout = QVBoxLayout(folder_group)
        
        self.folder_path = QLineEdit()
        self.folder_path.setPlaceholderText("选择包含船舶轨迹CSV的文件夹...")
        folder_btn = QPushButton("选择文件夹")
        folder_btn.clicked.connect(self.select_folder)
        
        folder_layout.addWidget(QLabel("数据文件夹路径:"))
        folder_layout.addWidget(self.folder_path)
        folder_layout.addWidget(folder_btn)
        control_layout.addWidget(folder_group)

        # 区域筛选
        filter_group = QGroupBox("区域筛选")
        filter_layout = QGridLayout(filter_group)
        
        self.min_lat = QLineEdit()
        self.max_lat = QLineEdit()
        self.min_lon = QLineEdit()
        self.max_lon = QLineEdit()
        
        filter_layout.addWidget(QLabel("最小纬度:"), 0, 0)
        filter_layout.addWidget(self.min_lat, 0, 1)
        filter_layout.addWidget(QLabel("最大纬度:"), 1, 0)
        filter_layout.addWidget(self.max_lat, 1, 1)
        filter_layout.addWidget(QLabel("最小经度:"), 2, 0)
        filter_layout.addWidget(self.min_lon, 2, 1)
        filter_layout.addWidget(QLabel("最大经度:"), 3, 0)
        filter_layout.addWidget(self.max_lon, 3, 1)
        
        filter_btn = QPushButton("筛选并绘制")
        filter_btn.clicked.connect(self.filter_and_plot)
        filter_layout.addWidget(filter_btn, 4, 0, 1, 2)
        
        control_layout.addWidget(filter_group)

        # 统计信息
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("数据未加载")
        self.stats_label.setStyleSheet("font-size: 11pt;")
        self.stats_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.stats_label.setWordWrap(True)
        
        stats_layout.addWidget(self.stats_label)
        control_layout.addWidget(stats_group)

        # 图例
        legend_group = QGroupBox("图例说明")
        legend_layout = QVBoxLayout(legend_group)
        
        legend = QLabel()
        legend.setPixmap(self.create_legend_image())
        legend.setAlignment(Qt.AlignCenter)
        
        legend_layout.addWidget(legend)
        control_layout.addWidget(legend_group)

        # 添加到主布局
        main_layout.addWidget(control_panel)

        # 右侧地图区域
        map_frame = QFrame()
        map_layout = QVBoxLayout(map_frame)
        map_layout.setContentsMargins(0, 0, 0, 0)
        
        # 地图控制工具栏
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        
        self.refresh_btn = QAction(QIcon(self.create_icon("refresh", "#1E90FF")), "刷新", self)
        self.refresh_btn.triggered.connect(self.refresh_map)
        toolbar.addAction(self.refresh_btn)
        
        self.zoom_in_btn = QAction(QIcon(self.create_icon("plus", "#1E90FF")), "放大", self)
        self.zoom_in_btn.triggered.connect(self.zoom_in)
        toolbar.addAction(self.zoom_in_btn)
        
        self.zoom_out_btn = QAction(QIcon(self.create_icon("minus", "#1E90FF")), "缩小", self)
        self.zoom_out_btn.triggered.connect(self.zoom_out)
        toolbar.addAction(self.zoom_out_btn)
        
        self.theme_btn = QAction(QIcon(self.create_icon("theme", "#1E90FF")), "切换主题", self)
        self.theme_btn.triggered.connect(self.toggle_theme)
        toolbar.addAction(self.theme_btn)
        
        map_layout.addWidget(toolbar)

        # Web视图显示地图
        self.web_view = QWebEngineView()
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.init_map()
        
        map_layout.addWidget(self.web_view)
        main_layout.addWidget(map_frame, 1)

    def create_h_line(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #2A2A2A;")
        return line

    def create_icon(self, icon_type, color):
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(color), 2))
        
        if icon_type == "refresh":
            painter.drawArc(4, 4, 16, 16, 0, 360 * 16)
            painter.drawLine(12, 4, 16, 0)
        elif icon_type == "plus":
            painter.drawLine(12, 4, 12, 20)
            painter.drawLine(4, 12, 20, 12)
        elif icon_type == "minus":
            painter.drawLine(4, 12, 20, 12)
        elif icon_type == "theme":
            painter.drawEllipse(4, 4, 16, 16)
            painter.drawLine(12, 12, 20, 20)
        
        painter.end()
        return pixmap

    def create_legend_image(self):
        # 创建图例图像
        image = QPixmap(300, 180)
        image.fill(Qt.transparent)
        
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Arial", 10))
        
        # 背景
        painter.setBrush(QColor(30, 30, 50, 200))
        painter.setPen(QPen(QColor(70, 130, 180), 1))
        painter.drawRoundedRect(0, 0, 300, 180, 5, 5)
        
        # 标题
        painter.setPen(QColor(30, 144, 255))
        painter.drawText(20, 30, "图例说明")
        
        # 内容
        colors = [
            ("货船", QColor(65, 105, 225)),
            ("油轮", QColor(50, 205, 50)),
            ("客船", QColor(255, 140, 0)),
            ("渔船", QColor(148, 0, 211)),
            ("军舰", QColor(220, 20, 60))
        ]
        
        y_pos = 50
        for name, color in colors:
            painter.setBrush(color)
            painter.drawEllipse(20, y_pos, 12, 12)
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(40, y_pos + 10, name)
            y_pos += 25
        
        painter.end()
        return image

    def apply_dark_theme(self):
        dark_stylesheet = """
        QWidget {
            background-color: #0A0A1A;
            color: #E0E0FF;
            font-family: 'Segoe UI';
        }
        QGroupBox {
            font-size: 12pt;
            color: #1E90FF;
            border: 1px solid #2A2A4A;
            border-radius: 5px;
            margin-top: 1ex;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
            background-color: #0A0A1A;
        }
        QLineEdit, QPushButton {
            background-color: #151530;
            color: #E0E0FF;
            border: 1px solid #2A2A4A;
            border-radius: 3px;
            padding: 5px;
        }
        QPushButton {
            background-color: #1E3A5F;
            min-height: 30px;
        }
        QPushButton:hover {
            background-color: #2A4A7F;
        }
        QToolBar {
            background-color: #101025;
            border: none;
            padding: 2px;
        }
        QWebEngineView {
            border: 1px solid #2A2A4A;
        }
        """
        self.setStyleSheet(dark_stylesheet)

    def init_map(self):
        """初始化地图"""
        self.current_map = folium.Map(
            location=[30.0, 120.0],
            zoom_start=8,
            tiles='CartoDB dark_matter',
            control_scale=True,
            attr='Marine Traffic Visualization'
        )
        self.update_map()

    def update_map(self):
        """更新地图显示"""
        if self.current_map:
            data = BytesIO()
            self.current_map.save(data, close_file=False)
            self.map_html = data.getvalue().decode()
            self.web_view.setHtml(self.map_html)

    def select_folder(self):
        """选择包含船舶轨迹CSV的文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择船舶轨迹数据文件夹")
        if folder:
            self.folder_path.setText(folder)
            self.load_ship_data(folder)
            self.plot_all_tracks()

    def load_ship_data(self, folder_path):
        """加载文件夹中的所有CSV文件"""
        self.ship_data = {}
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        if not csv_files:
            QMessageBox.warning(self, "警告", "未找到CSV文件！")
            return
        
        for file in csv_files:
            try:
                df = pd.read_csv(os.path.join(folder_path, file))
                if {'latitude', 'longitude', 'speed', 'heading', 'type'}.issubset(df.columns):
                    ship_id = file.split('.')[0]
                    self.ship_data[ship_id] = df
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        self.update_stats()

    def update_stats(self):
        """更新统计信息"""
        if not self.ship_data:
            self.stats_label.setText("数据未加载")
            return
        
        num_ships = len(self.ship_data)
        total_points = sum(len(df) for df in self.ship_data.values())
        
        ship_types = {}
        for df in self.ship_data.values():
            ship_type = df['type'].mode()[0] if 'type' in df.columns else '未知'
            ship_types[ship_type] = ship_types.get(ship_type, 0) + 1
        
        type_info = "\n".join([f"{k}: {v}艘" for k, v in ship_types.items()])
        
        stats_text = f"""
        <b>数据统计:</b>
        <br>船舶数量: <font color='#1E90FF'>{num_ships}</font> 艘
        <br>轨迹点总数: <font color='#1E90FF'>{total_points}</font> 个
        <br>
        <br><b>船舶类型分布:</b>
        <br>{type_info}
        """
        self.stats_label.setText(stats_text)

    def plot_all_tracks(self):
        """绘制所有船舶轨迹"""
        if not self.ship_data:
            return
        
        self.current_map = folium.Map(
            location=self.get_center(),
            zoom_start=8,
            tiles='CartoDB dark_matter',
            control_scale=True
        )
        
        # 创建聚类标记
        marker_cluster = MarkerCluster(name="船舶位置").add_to(self.current_map)
        
        # 船舶类型颜色映射
        color_map = {
            'cargo': '#4169E1',    # 货船 - 皇家蓝
            'tanker': '#32CD32',   # 油轮 - 酸橙绿
            'passenger': '#FF8C00', # 客船 - 深橙色
            'fishing': '#9400D3',  # 渔船 - 深紫罗兰
            'military': '#DC143C',  # 军舰 - 深红
            'default': '#1E90FF'    # 默认 - 道奇蓝
        }
        
        # 绘制每条轨迹
        for ship_id, df in self.ship_data.items():
            ship_type = df['type'].mode()[0] if 'type' in df.columns else 'default'
            color = color_map.get(ship_type.lower(), color_map['default'])
            
            # 创建轨迹线
            points = list(zip(df['latitude'], df['longitude']))
            folium.PolyLine(
                points,
                color=color,
                weight=2,
                opacity=0.7,
                tooltip=f"{ship_id} ({ship_type})"
            ).add_to(self.current_map)
            
            # 添加起点和终点标记
            if len(points) > 0:
                # 起点
                folium.Marker(
                    points[0],
                    icon=folium.Icon(color='green', icon='play', prefix='fa'),
                    tooltip=f"{ship_id} 起点"
                ).add_to(marker_cluster)
                
                # 终点
                folium.Marker(
                    points[-1],
                    icon=folium.Icon(color='red', icon='stop', prefix='fa'),
                    tooltip=f"{ship_id} 终点"
                ).add_to(marker_cluster)
        
        # 添加图层控制
        folium.LayerControl().add_to(self.current_map)
        self.update_map()

    def filter_and_plot(self):
        """根据输入的经纬度范围筛选并绘制船舶轨迹"""
        try:
            min_lat = float(self.min_lat.text())
            max_lat = float(self.max_lat.text())
            min_lon = float(self.min_lon.text())
            max_lon = float(self.max_lon.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的经纬度数值")
            return
        
        if not self.ship_data:
            QMessageBox.warning(self, "数据错误", "请先加载船舶数据")
            return
        
        # 创建筛选区域多边形
        area_polygon = Polygon([
            (min_lat, min_lon),
            (min_lat, max_lon),
            (max_lat, max_lon),
            (max_lat, min_lon)
        ])
        
        # 筛选轨迹
        filtered_data = {}
        for ship_id, df in self.ship_data.items():
            # 检查是否有轨迹点位于区域内
            in_area = False
            for _, row in df.iterrows():
                point = Point(row['latitude'], row['longitude'])
                if point.within(area_polygon):
                    in_area = True
                    break
            if in_area:
                filtered_data[ship_id] = df
        
        if not filtered_data:
            QMessageBox.information(self, "筛选结果", "该区域内未发现船舶轨迹")
            return
        
        # 更新地图
        self.current_map = folium.Map(
            location=[(min_lat + max_lat)/2, (min_lon + max_lon)/2],
            zoom_start=10,
            tiles='CartoDB dark_matter',
            control_scale=True
        )
        
        # 绘制筛选区域
        folium.Rectangle(
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            color='#FF4500',
            fill=True,
            fill_color='#FF4500',
            fill_opacity=0.1,
            weight=2,
            tooltip="筛选区域"
        ).add_to(self.current_map)
        
        # 船舶类型颜色映射
        color_map = {
            'cargo': '#4169E1',
            'tanker': '#32CD32',
            'passenger': '#FF8C00',
            'fishing': '#9400D3',
            'military': '#DC143C',
            'default': '#1E90FF'
        }
        
        # 绘制筛选后的轨迹
        for ship_id, df in filtered_data.items():
            ship_type = df['type'].mode()[0] if 'type' in df.columns else 'default'
            color = color_map.get(ship_type.lower(), color_map['default'])
            
            points = list(zip(df['latitude'], df['longitude']))
            folium.PolyLine(
                points,
                color=color,
                weight=3,
                opacity=0.8,
                tooltip=f"{ship_id} ({ship_type})"
            ).add_to(self.current_map)
            
            # 添加起点和终点标记
            if len(points) > 0:
                folium.Marker(
                    points[0],
                    icon=folium.Icon(color='green', icon='play', prefix='fa'),
                    tooltip=f"{ship_id} 起点"
                ).add_to(self.current_map)
                
                folium.Marker(
                    points[-1],
                    icon=folium.Icon(color='red', icon='stop', prefix='fa'),
                    tooltip=f"{ship_id} 终点"
                ).add_to(self.current_map)
        
        self.update_map()
        
        # 更新统计信息
        num_ships = len(filtered_data)
        total_points = sum(len(df) for df in filtered_data.values())
        self.stats_label.setText(
            f"<b>筛选结果:</b><br>"
            f"区域内船舶数量: <font color='#1E90FF'>{num_ships}</font> 艘<br>"
            f"轨迹点总数: <font color='#1E90FF'>{total_points}</font> 个"
        )

    def get_center(self):
        """计算所有轨迹的中心点"""
        if not self.ship_data:
            return [30.0, 120.0]  # 默认位置（中国东海附近）
        
        all_lats = []
        all_lons = []
        
        for df in self.ship_data.values():
            if not df.empty:
                all_lats.extend(df['latitude'].tolist())
                all_lons.extend(df['longitude'].tolist())
        
        if not all_lats:
            return [30.0, 120.0]
        
        avg_lat = sum(all_lats) / len(all_lats)
        avg_lon = sum(all_lons) / len(all_lons)
        return [avg_lat, avg_lon]

    def refresh_map(self):
        """刷新地图，重新绘制所有轨迹"""
        if self.ship_data:
            self.plot_all_tracks()

    def zoom_in(self):
        """地图放大"""
        self.web_view.page().runJavaScript("map.setZoom(map.getZoom()+1);")

    def zoom_out(self):
        """地图缩小"""
        self.web_view.page().runJavaScript("map.setZoom(map.getZoom()-1);")

    def toggle_theme(self):
        """切换地图主题"""
        if self.dark_theme:
            tiles = 'OpenStreetMap'
        else:
            tiles = 'CartoDB dark_matter'
        
        self.web_view.page().runJavaScript(f"map.eachLayer(function(layer){{map.removeLayer(layer);}});")
        self.web_view.page().runJavaScript(f"""
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }}).addTo(map);
        """ if not self.dark_theme else f"""
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                subdomains: 'abcd',
                maxZoom: 20
            }}).addTo(map);
        """)
        
        self.dark_theme = not self.dark_theme

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 设置全局字体
    font = QFont("Microsoft YaHei UI", 10)
    app.setFont(font)
    
    window = ShipTrackVisualizer()
    window.show()
    sys.exit(app.exec_())