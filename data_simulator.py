import numpy as np
import scipy.io
import socketio
import time
import random

# SocketIO 客户端
sio = socketio.Client()

# 连接主程序SocketIO服务
SERVER_URL = 'http://localhost:5000'  # 根据实际端口调整

# 加载数据
normal = scipy.io.loadmat('97.mat')["X097_DE_time"].flatten()
fault = scipy.io.loadmat('105.mat')["X105_DE_time"].flatten()


# 数据片段长度
SEQ_LEN = 200
MACHINE_NUM = 3  # 机床数量

# 连接事件
@sio.event
def connect():
    print('已连接到主程序 SocketIO 服务')

@sio.event
def disconnect():
    print('与主程序断开连接')

def send_data(machine_id, label='normal'):
    """
    machine_id: 机床编号（int）
    label: 'normal' 或 'fault'
    """
    if label == 'normal':
        arr = normal
    else:
        arr = fault
    # 随机取一段
    start = random.randint(0, len(arr) - SEQ_LEN)
    segment = arr[start:start+SEQ_LEN].tolist()
    # 发送到主程序，包含机床编号
    sio.emit('simulated_signal', {'machine_id': machine_id, 'signal': segment, 'label': label})
    print(f'机床{machine_id} 已发送{label}数据片段')

if __name__ == '__main__':
    sio.connect(SERVER_URL)
    try:
        while True:
            # 每台机床独立随机发送
            for machine_id in range(1, MACHINE_NUM+1):
                label = random.choice(['normal', 'fault'])
                send_data(machine_id, label)
            time.sleep(5)  # 每5秒所有机床各发一次
    except KeyboardInterrupt:
        print('退出模拟器')
    finally:
        sio.disconnect()
