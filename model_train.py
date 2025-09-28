import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 加载正常数据
mat_normal = scipy.io.loadmat('97.mat')
# 数据在一个名为 'X097_DE_time' 的键中
normal_data = mat_normal['X097_DE_time'].flatten()

# 加载故障数据
mat_fault = scipy.io.loadmat('105.mat')
# 数据在一个名为 'X105_DE_time' 的键中
fault_data = mat_fault['X105_DE_time'].flatten()

# 可视化对比
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(normal_data[:2000]) # 只看前2000个点
plt.title('Normal Vibration Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(1, 2, 2)
plt.plot(fault_data[:2000])
plt.title('Faulty Vibration Signal')
plt.xlabel('Time')
plt.show()

# 为了简化，我们只取一小段数据来做演示
data_len = 48000
normal_slice = normal_data[:data_len]
fault_slice = fault_data[:data_len]

# 创建DataFrame
df_normal = pd.DataFrame({'signal': normal_slice, 'label': 0})
df_fault = pd.DataFrame({'signal': fault_slice, 'label': 1})

# 合并数据
df = pd.concat([df_normal, df_fault], ignore_index=True)
print("Combined DataFrame head:")
print(df.head())
print("\nCombined DataFrame tail:")
print(df.tail())

# 先归一化信号数据
scaler = MinMaxScaler()
df['scaled_signal'] = scaler.fit_transform(df[['signal']])

# 定义一个函数来创建序列
def create_sequences(data, labels, time_steps=100):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        # 标签是这个窗口结束后下一个点的状态
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 200 # 每个窗口的长度
X, y = create_sequences(df['scaled_signal'].values, df['label'].values, time_steps)

# LSTM需要三维输入 [样本数, 时间步长, 特征数]
X = X.reshape(X.shape[0], X.shape[1], 1)

print(f"Shape of X: {X.shape}") # 应该输出 (..., 200, 1)
print(f"Shape of y: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # stratify保证训练集和测试集中0和1的比例一致
)
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. 构建模型结构
model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2)) # Dropout层防止过拟合
model.add(Dense(units=1, activation='sigmoid')) # 输出层，sigmoid输出0-1之间的概率

# 2. 编译模型：告诉模型如何学习
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. 显示模型结构，可以截图放入PPT
model.summary()

# 4. 开始训练！
# verbose=1 会显示训练进度条
history = model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.2, verbose=1)




loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss:.4f}")


# 定义一个文件名来保存你的模型
model_filename = 'cnc_fault_predictor.keras'

# 使用 model.save() 函数保存模型
model.save(model_filename)

print(f"模型已成功保存到文件: {model_filename}")


plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy during Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



