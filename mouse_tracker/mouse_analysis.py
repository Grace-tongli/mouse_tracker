# mouse_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pynput import mouse
import csv
import os
import threading
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from antropy import app_entropy  # 关键库替换
import matplotlib.font_manager as fm


# 设置中文字体支持
def set_chinese_font():
    """自动检测并设置可用的中文字体"""
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei"]

    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            print(f"已设置中文字体: {font}")
            return

    print("未找到可用的中文字体，图表中的中文可能无法正确显示")


set_chinese_font()


# ====================== 数据采集模块 ======================
class MouseTracker:
    def __init__(self):
        self.current_x, self.current_y = 0, 0
        self.last_x, self.last_y = None, None
        self.last_time = time.time()
        self.data_buffer = []
        self.is_running = True
        self.file_path = "mouse_data.csv"
        self._init_csv()

    def _init_csv(self):
        """初始化CSV文件"""
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "x", "y", "event_type", "speed", "acceleration"])

    def _calculate_speed_acceleration(self, x1, y1, t1, x2, y2, t2):
        """计算速度和加速度"""
        delta_time = (t2 - t1) * 1000  # 毫秒
        if delta_time == 0:
            return 0, 0
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        speed = distance / delta_time  # 像素/毫秒
        acceleration = (speed / delta_time) if delta_time != 0 else 0
        return speed, acceleration

    def _save_to_csv(self):
        """定时保存数据"""
        while self.is_running:
            if self.data_buffer:
                with open(self.file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(self.data_buffer)
                    self.data_buffer = []
            time.sleep(1)

    def on_move(self, x, y):
        """鼠标移动回调"""
        event_time = time.time()
        speed, acceleration = 0, 0
        if self.last_x is not None:
            speed, acceleration = self._calculate_speed_acceleration(
                self.last_x, self.last_y, self.last_time, x, y, event_time
            )
        # 强制转换为整数并保存
        self.data_buffer.append([
            int(event_time * 1000),
            int(x),  # 确保x为整数
            int(y),  # 确保y为整数
            "move",
            round(speed, 2),
            round(acceleration, 2)
        ])
        self.last_x, self.last_y = x, y
        self.last_time = event_time

    def on_click(self, x, y, button, pressed):
        """鼠标点击回调"""
        event_type = f"{'press' if pressed else 'release'}_{button.name}"
        self.data_buffer.append([
            int(time.time() * 1000),
            int(x),  # 确保x为整数
            int(y),  # 确保y为整数
            event_type,
            0, 0  # 点击事件速度和加速度为0
        ])

    def start(self):
        """启动监听"""
        save_thread = threading.Thread(target=self._save_to_csv)
        save_thread.daemon = True
        save_thread.start()
        with mouse.Listener(on_move=self.on_move, on_click=self.on_click) as listener:
            print("鼠标监听已启动，按 Esc 键停止...")
            listener.join()


# ====================== 数据分析模块 ======================
class MouseAnalyzer:
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height

    def load_and_clean(self, file_path):
        """加载并清洗数据"""
        df = pd.read_csv(file_path)
        # 清理列名空格
        df.columns = df.columns.str.strip()
        # 强制类型转换
        df['x'] = pd.to_numeric(df['x'], errors='coerce').fillna(0).astype(int)
        df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0).astype(int)
        # 确保 speed 列是数值类型
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce').fillna(0)
        # 过滤掉 timestamp 列中无法转换为数值类型的行
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        # 过滤无效坐标
        df = df[
            (df['x'].between(0, self.screen_width)) &
            (df['y'].between(0, self.screen_height)) &
            (df['speed'] >= 0)
            ]
        # 时间处理
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df['time_seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        # 坐标归一化
        df['x_norm'] = df['x'] / self.screen_width
        df['y_norm'] = df['y'] / self.screen_height
        return df

    def extract_features(self, df, window_size=10):
        """提取时间窗口特征"""
        # 将 time_seconds 转换为 Timedelta 类型
        df['time_seconds'] = pd.to_timedelta(df['time_seconds'], unit='s')

        # 将 event_type 转换为数值类型（是否为 move）
        df['is_move'] = df['event_type'] == 'move'

        # 设置 time_seconds 为索引
        df = df.set_index('time_seconds')

        # 使用时间窗口进行滚动计算
        rolling = df.rolling(f"{window_size}s")
        feature_df = rolling.agg({
            'speed': ['mean', 'std'],
            'acceleration': 'max',
            'is_move': 'mean'  # 计算 move 事件的频率
        })
        feature_df.columns = ['avg_speed', 'speed_std', 'max_accel', 'move_freq']
        feature_df = feature_df.reset_index()

        # 点击频率
        clicks = df[df['event_type'].str.contains('click')]
        click_counts = clicks.groupby(
            pd.Grouper(freq=f"{window_size}s")
        ).size().reset_index(name='clicks')
        click_counts['time_seconds'] = pd.to_timedelta(click_counts['time_seconds'])
        feature_df = pd.merge_asof(feature_df, click_counts, on='time_seconds')

        # 轨迹复杂度 (使用antropy)
        move_data = df[df['event_type'] == 'move'].copy()
        if not move_data.empty:
            feature_df['x_entropy'] = app_entropy(
                move_data['x_norm'].values,
                order=2,
                metric='chebyshev'
            )
        return feature_df

    def visualize(self, df, feature_df):
        """可视化分析"""
        # 轨迹热力图
        plt.figure(figsize=(12, 6))
        sns.kdeplot(x=df['x_norm'], y=df['y_norm'], cmap='viridis', fill=True)
        plt.title("鼠标轨迹停留密度热力图")
        plt.savefig('heatmap.png')
        print("热力图已保存至:", os.path.abspath('heatmap.png'))
        plt.close()

        # 速度-点击频率时序图
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(feature_df['time_seconds'].dt.total_seconds(), feature_df['avg_speed'], 'b-', label='平均速度')
        ax2 = ax1.twinx()
        ax2.plot(feature_df['time_seconds'].dt.total_seconds(), feature_df['clicks'], 'r--', label='点击次数')
        plt.title("速度与点击行为随时间变化")
        fig.legend(loc='upper right')
        plt.savefig('speed_vs_clicks.png')
        print("速度与点击行为时序图已保存至:", os.path.abspath('speed_vs_clicks.png'))
        plt.close()

        # 鼠标点击位置分布图
        click_df = df[df['event_type'].str.contains('click')]
        plt.figure(figsize=(12, 6))
        plt.scatter(click_df['x_norm'], click_df['y_norm'], alpha=0.5, c='blue')
        plt.title("鼠标点击位置分布图")
        plt.xlabel("归一化X坐标")
        plt.ylabel("归一化Y坐标")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('click_distribution.png')
        print("点击位置分布图已保存至:", os.path.abspath('click_distribution.png'))
        plt.close()

        # 速度分布图
        move_df = df[df['event_type'] == 'move']
        plt.figure(figsize=(10, 6))
        sns.histplot(move_df['speed'], kde=True, bins=50)
        plt.title("鼠标移动速度分布")
        plt.xlabel("速度 (像素/毫秒)")
        plt.ylabel("频次")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('speed_distribution.png')
        print("速度分布图已保存至:", os.path.abspath('speed_distribution.png'))
        plt.close()

    def train_model(self, feature_df):
        """训练情绪预测模型（示例模拟标签）"""
        # 确保用于生成标签的特征列不包含 NaN
        feature_df['speed_std'] = feature_df['speed_std'].fillna(0)
        feature_df['clicks'] = feature_df['clicks'].fillna(0)

        # 生成模拟情绪标签
        np.random.seed(42)
        feature_df['stress_level'] = np.clip(
            feature_df['speed_std'] * 0.8 +
            feature_df['clicks'] * 0.3 +
            np.random.normal(0, 0.5, len(feature_df)),
            0, 10
        )

        # 特征选择
        X = feature_df[['avg_speed', 'speed_std', 'clicks', 'x_entropy']].fillna(0)
        y = feature_df['stress_level']

        # 确认没有 NaN
        if y.isna().any():
            print(f"警告: 目标变量 y 中包含 {y.isna().sum()} 个 NaN 值，将其替换为 0")
            y = y.fillna(0)

        # 训练模型
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)

        # 评估
        y_pred = model.predict(X_test)
        print(f"模型性能:\nMAE: {mean_absolute_error(y_test, y_pred):.2f}\nR²: {r2_score(y_test, y_pred):.2f}")

        # 特征重要性
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        feat_importances.sort_values().plot(kind='barh')
        plt.title("情绪预测特征重要性")
        plt.savefig('feature_importance.png')
        print("特征重要性图已保存至:", os.path.abspath('feature_importance.png'))
        plt.close()


# ====================== 主程序 ======================
if __name__ == "__main__":
    # Step 1: 采集数据
    if not os.path.exists("mouse_data.csv"):
        print("启动鼠标数据采集...")
        tracker = MouseTracker()
        try:
            tracker.start()
        except KeyboardInterrupt:
            tracker.is_running = False
            print("\n数据已保存至 mouse_data.csv")

    # Step 2: 分析数据
    analyzer = MouseAnalyzer(screen_width=1920, screen_height=1080)
    df = analyzer.load_and_clean("mouse_data.csv")
    feature_df = analyzer.extract_features(df)
    analyzer.visualize(df, feature_df)
    analyzer.train_model(feature_df)
    print("分析完成！所有图表已保存")