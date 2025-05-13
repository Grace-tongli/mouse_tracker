# mouse_tracker.py
from pynput import mouse
import time
import csv
import os
import threading

# 全局变量初始化
current_x, current_y = 0, 0
last_x, last_y = None, None
last_time = time.time()
data_buffer = []
is_running = True

# CSV 文件头
CSV_HEADER = ["timestamp", "x", "y", "event_type", "speed", "acceleration"]
FILE_PATH = "mouse_data.csv"


# 创建并初始化 CSV 文件
def init_csv():
    if not os.path.exists(FILE_PATH):
        with open(FILE_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


# 计算速度和加速度
def calculate_speed_acceleration(x1, y1, t1, x2, y2, t2):
    delta_time = (t2 - t1) * 1000  # 毫秒
    if delta_time == 0:
        return 0, 0
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    speed = distance / delta_time  # 像素/毫秒
    acceleration = speed / delta_time if delta_time != 0 else 0
    return speed, acceleration


# 定时保存数据到 CSV（防止频繁IO）
def save_to_csv():
    global data_buffer
    while is_running:
        if len(data_buffer) > 0:
            with open(FILE_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(data_buffer)
                data_buffer = []
        time.sleep(1)  # 每1秒保存一次


# 鼠标移动监听回调
def on_move(x, y):
    global last_x, last_y, last_time, current_x, current_y
    current_x, current_y = x, y
    event_time = time.time()

    # 计算速度和加速度
    speed, acceleration = 0, 0
    if last_x is not None:
        speed, acceleration = calculate_speed_acceleration(
            last_x, last_y, last_time, x, y, event_time
        )

    # 记录数据到缓冲区
    data_buffer.append([
        int(event_time * 1000),  # 毫秒级时间戳
        x, y,
        "move",
        round(speed, 2),
        round(acceleration, 2)
    ])

    # 更新上一次的位置和时间
    last_x, last_y = x, y
    last_time = event_time


# 鼠标点击监听回调
def on_click(x, y, button, pressed):
    event_type = f"{'press' if pressed else 'release'}_{button.name}"
    data_buffer.append([
        int(time.time() * 1000),
        x, y,
        event_type,
        0, 0  # 点击事件速度和加速度为0
    ])


# 主程序
def main():
    init_csv()

    # 启动数据保存线程
    save_thread = threading.Thread(target=save_to_csv)
    save_thread.daemon = True
    save_thread.start()

    # 监听鼠标事件
    with mouse.Listener(on_move=on_move, on_click=on_click) as listener:
        print("鼠标监听已启动，按 Esc 键停止...")
        listener.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        is_running = False
        print("\n监听已停止，数据保存至 mouse_data.csv")