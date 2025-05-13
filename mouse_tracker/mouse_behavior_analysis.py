import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ======================
# 1. 数据解析与结构理解
# ======================
# 定义列名并读取数据
columns = ["timestamp", "x", "y", "action", "value1", "value2"]
df = pd.read_csv("mouse_data.csv", names=columns)

# 转换时间戳为可读格式
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# ======================
# 2. 数据清洗与预处理
# ======================
# 检查缺失值
print("缺失值统计:")
print(df.isnull().sum())

# 按时间排序
df = df.sort_values("timestamp").reset_index(drop=True)

# 验证坐标范围
print("\n坐标范围:")
print(f"X: [{df['x'].min()}, {df['x'].max()}]")
print(f"Y: [{df['y'].min()}, {df['y'].max()}]")

# ======================
# 3. 事件分类与过滤
# ======================
# 分离不同类型事件
move_events = df[df["action"] == "move"]
click_press = df[df["action"] == "press_left"]
click_release = df[df["action"] == "release_left"]

# 点击事件配对（简单示例）
clicks = []
for idx, press in click_press.iterrows():
    release = click_release[click_release["timestamp"] > press["timestamp"]].head(1)
    if not release.empty:
        duration = (release["timestamp"].iloc[0] - press["timestamp"]).total_seconds()
        clicks.append({
            "press_x": press["x"],
            "press_y": press["y"],
            "duration": duration
        })
click_df = pd.DataFrame(clicks)

# ======================
# 4. 基础可视化：移动轨迹
# ======================
plt.figure(figsize=(15, 8))
plt.scatter(move_events["x"], move_events["y"],
           s=1, c="blue", alpha=0.5, label="Movement")
if not click_df.empty:
    plt.scatter(click_df["press_x"], click_df["press_y"],
               s=50, c="red", marker="X", label="Clicks")
plt.title("Mouse Movement Trajectory with Click Events")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.gca().invert_yaxis()  # 反转Y轴（适用于屏幕坐标系）
plt.legend()
plt.grid(True)
plt.savefig("movement_trajectory.png", dpi=300)
plt.close()

# ======================
# 5. 移动速度分析
# ======================
# 计算时间差（秒）和位移
df["time_diff"] = df["timestamp"].diff().dt.total_seconds()
df["dx"] = df["x"].diff().abs()
df["dy"] = df["y"].diff().abs()
df["distance"] = (df["dx"]**2 + df["dy"]**2)**0.5
df["speed"] = df["distance"] / df["time_diff"]

# 过滤异常速度（如首次计算的NaN或极大值）
df = df[(df["speed"] < 100) & (df["speed"].notnull())]

# 速度分布直方图
plt.figure(figsize=(12, 6))
plt.hist(df["speed"], bins=50, color="purple", alpha=0.7)
plt.title("Mouse Speed Distribution")
plt.xlabel("Speed (pixels/second)")
plt.ylabel("Frequency")
plt.savefig("speed_distribution.png", dpi=300)
plt.close()

# ======================
# 6. 点击事件分析
# ======================
if not click_df.empty:
    # 点击热力图
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=click_df["press_x"], y=click_df["press_y"],
                cmap="Reds", shade=True, bw_adjust=0.5)
    plt.title("Click Heatmap")
    plt.gca().invert_yaxis()
    plt.savefig("click_heatmap.png", dpi=300)
    plt.close()

    # 点击时长分析
    plt.figure(figsize=(10, 4))
    plt.hist(click_df["duration"], bins=20, color="green", alpha=0.7)
    plt.title("Click Duration Distribution")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Count")
    plt.savefig("click_duration.png", dpi=300)
    plt.close()

# ======================
# 7. 时间序列分析
# ======================
# 按分钟统计事件频率
df.set_index("timestamp", inplace=True)
events_per_min = df.resample("1min").size()

plt.figure(figsize=(12, 4))
events_per_min.plot(color="teal")
plt.title("Event Frequency Over Time")
plt.ylabel("Events per Minute")
plt.savefig("event_frequency.png", dpi=300)
plt.close()

# ======================
# 8. 生成报告
# ======================
report = f"""
=== 分析报告 ===
总数据量: {len(df):,} 条
移动事件: {len(move_events):,} 条
点击事件: {len(click_df):,} 次
平均移动速度: {df["speed"].mean():.2f} 像素/秒
最大移动速度: {df["speed"].max():.2f} 像素/秒
"""

if not click_df.empty:
    report += f"""
点击相关统计:
- 平均点击时长: {click_df["duration"].mean():.3f} 秒
- 最久点击时长: {click_df["duration"].max():.3f} 秒
- 点击热点区域: X({click_df["press_x"].mean():.0f}±{click_df["press_x"].std():.0f}), 
               Y({click_df["press_y"].mean():.0f}±{click_df["press_y"].std():.0f})
"""

print(report)
with open("analysis_report.txt", "w") as f:
    f.write(report)