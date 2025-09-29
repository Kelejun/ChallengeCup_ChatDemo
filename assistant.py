import websocket
import hashlib
import base64
import hmac
import json
import time
import pyttsx3
from datetime import datetime, timezone
from urllib.parse import urlencode
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import asyncio

# ========== 故障预测模型相关 ========== 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def initial_predict():
    global latest_sim_predict
    try:
        import scipy.io
        normal = scipy.io.loadmat('97.mat')["X097_DE_time"].flatten()
        fault = scipy.io.loadmat('105.mat')["X105_DE_time"].flatten()
        for machine_id in range(1, 4):
            import random
            label = random.choice(['normal', 'fault'])
            arr = normal if label == 'normal' else fault
            start = random.randint(0, len(arr) - 200)
            segment = arr[start:start+200]
            sig_arr = np.array(segment).reshape(-1, 1)
            sig_scaled = scaler.transform(sig_arr).flatten()
            seq = sig_scaled[-200:]
            X = seq.reshape(1, 200, 1)
            pred = fault_model.predict(X)
            prob = float(pred[0][0])
            pred_label = 1 if prob > 0.5 else 0
            msg = f"机床{machine_id}初始推理：{'故障' if pred_label==1 else '正常'} (概率 {prob:.2f})，模拟标签：{label}"
            latest_sim_predict[machine_id] = msg
            print(f'机床{machine_id}初始推理完成，结果：{msg}')
    except Exception as e:
        print(f'初始推理异常: {e}')

# 加载模型和归一化器（全局只加载一次）
try:
    fault_model = load_model('cnc_fault_predictor.keras')
    scaler = MinMaxScaler()
    # 这里假设模型训练时用的归一化器参数与训练脚本一致，实际可保存scaler参数，或用训练数据初始化
    # 先用正常+故障样本初始化scaler（如有需要可优化）
    import scipy.io
    normal = scipy.io.loadmat('97.mat')["X097_DE_time"].flatten()[:48000]
    fault = scipy.io.loadmat('105.mat')["X105_DE_time"].flatten()[:48000]
    scaler.fit(np.concatenate([normal, fault]).reshape(-1, 1))
    print("故障预测模型和归一化器加载成功")
except Exception as e:
    fault_model = None
    scaler = None
    print(f"故障预测模型加载失败: {e}")



# ========== 配置区：通义千问Plus & 阿里云ASR API Key ==========
QWEN_API_KEY = ""  # 通义千问API Key
ALI_ASR_API_KEY = ""  # 阿里云百炼API Key（必填）
ALI_ASR_APPKEY = ""   # 阿里云ASR服务appkey（必填）

# ========== 导入通义千问API和ASR封装 ========== 
from qwen_api import get_qwen_response
from ali_asr import ali_asr_recognize

# ========== Flask Web 应用 ========== 
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# ========== 语音识别SocketIO事件 ==========
@socketio.on('asr_audio')
def handle_asr_audio(data):
    """
    data: { audio: base64字符串（不含data:前缀） }
    """
    audio_base64 = data.get('audio', None)
    if not audio_base64:
        socketio.emit('log_message', {'message': '[ASR] 未收到音频数据'})
        emit('asr_result', {'text': '', 'error': '未收到音频数据'})
        return
    socketio.emit('log_message', {'message': f'[ASR] 收到音频数据，长度: {len(audio_base64)} 字节'})
    if not ALI_ASR_API_KEY or not ALI_ASR_APPKEY:
        socketio.emit('log_message', {'message': '[ASR] ASR配置缺失，请检查API Key和AppKey'})
        emit('asr_result', {'text': '', 'error': 'ASR配置缺失'})
        return
    try:
        text = ali_asr_recognize(audio_base64, apikey=ALI_ASR_API_KEY, appkey=ALI_ASR_APPKEY)
        if text:
            socketio.emit('log_message', {'message': f'[ASR] 识别成功: {text}'})
        else:
            socketio.emit('log_message', {'message': '[ASR] 识别完成，但未返回文本'})
        emit('asr_result', {'text': text})
    except Exception as e:
        socketio.emit('log_message', {'message': f'[ASR] 识别异常: {e}'})
        emit('asr_result', {'text': '', 'error': str(e)})

# ========== 导入通义千问API封装 ==========
from qwen_api import get_qwen_response

# ========== 初始化语音引擎 ==========
engine = pyttsx3.init()
engine.setProperty('rate', 180)      # 语速
engine.setProperty('volume', 0.9)    # 音量

# 全局变量用于接收响应
response_data = ""
received_response = False


conversation_history = []
# 存储最近一次模拟推送的预测结果
latest_sim_predict = {}  # {machine_id: 推理结果字符串}

# ========== Flask Web 应用 ==========
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# ========== 接收模拟信号并自动预测 ==========
@socketio.on('simulated_signal')
def handle_simulated_signal(data):
    """
    data: { 'machine_id': int, 'signal': [...], 'label': 'normal'|'fault' }
    """
    global latest_sim_predict
    machine_id = data.get('machine_id', None)
    signal = data.get('signal', None)
    label = data.get('label', '')
    socketio.emit('log_message', {'message': f"收到机床{machine_id}推送数据，标签：{label}，长度：{len(signal) if signal else 0}"})
    if fault_model is None or scaler is None:
        socketio.emit('log_message', {'message': '模型未加载，无法预测'})
        emit('fault_result', {'result': '模型未加载，无法预测'})
        return
    try:
        if signal is None or not isinstance(signal, list) or len(signal) < 200:
            socketio.emit('log_message', {'message': f'机床{machine_id}数据长度不足200，跳过'})
            emit('fault_result', {'result': '模拟数据长度不足200'})
            return
        socketio.emit('log_message', {'message': f'机床{machine_id}数据归一化并准备推理...'})
        sig_arr = np.array(signal).reshape(-1, 1)
        sig_scaled = scaler.transform(sig_arr).flatten()
        seq = sig_scaled[-200:]
        X = seq.reshape(1, 200, 1)
        socketio.emit('log_message', {'message': f'机床{machine_id}推理中...'})
        pred = fault_model.predict(X)
        prob = float(pred[0][0])
        pred_label = 1 if prob > 0.5 else 0
        msg = f"机床{machine_id}模拟推送数据预测：{'故障' if pred_label==1 else '正常'} (概率 {prob:.2f})，真实标签：{label}"
        latest_sim_predict[0] = msg
        socketio.emit('log_message', {'message': f'机床{machine_id}推理完成，结果：{msg}'})
        emit('sim_predict_result', {'result': msg, 'prob': prob, 'label': int(pred_label), 'true_label': label, 'machine_id': machine_id})
    except Exception as e:
        socketio.emit('log_message', {'message': f'机床{machine_id}推理异常: {e}'})
        emit('sim_predict_result', {'result': f'模拟数据预测出错: {e}', 'machine_id': machine_id})

# ========== 接收模拟信号并自动预测 ==========
@socketio.on('simulated_signal')
def handle_simulated_signal(data):
    """
    data: { 'signal': [...], 'label': 'normal'|'fault' }
    """
    global latest_sim_predict
    if fault_model is None or scaler is None:
        emit('fault_result', {'result': '模型未加载，无法预测'})
        return
    try:
        signal = data.get('signal', None)
        label = data.get('label', '')
        if signal is None or not isinstance(signal, list) or len(signal) < 200:
            emit('fault_result', {'result': '模拟数据长度不足200'})
            return
        sig_arr = np.array(signal).reshape(-1, 1)
        sig_scaled = scaler.transform(sig_arr).flatten()
        seq = sig_scaled[-200:]
        X = seq.reshape(1, 200, 1)
        pred = fault_model.predict(X)
        prob = float(pred[0][0])
        pred_label = 1 if prob > 0.5 else 0
        msg = f"模拟推送数据预测：{'故障' if pred_label==1 else '正常'} (概率 {prob:.2f})，真实标签：{label}"
        latest_sim_predict[0] = msg
        emit('sim_predict_result', {'result': msg, 'prob': prob, 'label': int(pred_label), 'true_label': label})
    except Exception as e:
        emit('sim_predict_result', {'result': f'模拟数据预测出错: {e}'})

# 创建一个全局事件循环用于TTS
tts_loop = None
tts_thread = None

def init_tts_loop():
    """初始化TTS事件循环"""
    global tts_loop, tts_thread
    tts_loop = asyncio.new_event_loop()
    tts_thread = threading.Thread(target=start_tts_loop, daemon=True)
    tts_thread.start()

def start_tts_loop():
    """在新线程中运行TTS事件循环"""
    asyncio.set_event_loop(tts_loop)
    tts_loop.run_forever()


# ========== 通义千问Plus对话API ==========
def get_llm_response(question):
    global conversation_history
    # 汇总所有机床的最新推理结果
    if latest_sim_predict:
        sim_info = "\n【各机床最新实时信号预测】\n" + "\n".join([
            f"机床{mid}: {info}" for mid, info in sorted(latest_sim_predict.items())
        ])
    else:
        sim_info = ""
    prompt = f"""你是一个智能数控机床运维专家，具备机床故障预测AI能力。
你可以调用本地LSTM模型对用户上传的时序信号数据进行分析，判断机床当前状态为“正常”或“故障”，并给出概率。
如有实时信号预测结果，请主动告知用户。
请用*简洁、专业、尽量口语化*的中文回答用户问题，并给出必要的建议，如及时安排检修等。{sim_info}
\n用户问题：{question}
"""
    conversation_history.append({"role": "user", "content": prompt})
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    try:
        reply = get_qwen_response(question=prompt, history=conversation_history[:-1], apikey=QWEN_API_KEY)
        conversation_history.append({"role": "assistant", "content": reply})
        socketio.emit('ai_response', {'message': reply})
        return reply
    except Exception as e:
        err = f"[通义千问API调用失败] {e}"
        socketio.emit('ai_response', {'message': err})
        return err


# ========== 文本转语音函数 ==========
def speak_text(text):
    if text and text.strip():
        print(f"AI正在说: {text}")
        socketio.emit('log_message', {'message': f"AI正在说: {text}"})
        # 使用自定义事件循环来避免事件循环冲突
        def _speak():
            engine.say(text)
            engine.runAndWait()
        
        # 将任务提交到专门的TTS事件循环中执行
        # 修改: 直接使用已创建的tts_loop，避免在子线程中获取事件循环
        if tts_loop and threading.active_count() > 1:  # 检查是否有活跃的线程
            # 使用 call_soon_threadsafe 在 TTS 线程中执行语音播放
            try:
                future = asyncio.run_coroutine_threadsafe(
                    asyncio.sleep(0), tts_loop  # 使用 run_coroutine_threadsafe
                )
                # 等待事件循环执行完毕
                future.result()
                # 在事件循环中执行语音
                tts_loop.call_soon_threadsafe(_speak)
            except Exception as e:
                print(f"异步执行出错: {e}")
                # 如果异步执行失败，则直接在当前线程播放
                _speak()
        else:
            # 如果 tts_loop 还未初始化或线程已死，则在当前线程直接播放
            _speak()
    else:
        print("没有内容可朗读")
        socketio.emit('log_message', {'message': "没有内容可朗读"})


# ========== Flask 路由 ==========
@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('send_message')
def handle_message(data):
    global latest_sim_predict
    # 对话时自动推理所有机床
    try:
        import scipy.io
        normal = scipy.io.loadmat('97.mat')["X097_DE_time"].flatten()
        fault = scipy.io.loadmat('105.mat')["X105_DE_time"].flatten()
        for machine_id in range(1, 4):
            import random
            label = random.choice(['normal', 'fault'])
            arr = normal if label == 'normal' else fault
            start = random.randint(0, len(arr) - 200)
            segment = arr[start:start+200]
            sig_arr = np.array(segment).reshape(-1, 1)
            sig_scaled = scaler.transform(sig_arr).flatten()
            seq = sig_scaled[-200:]
            X = seq.reshape(1, 200, 1)
            pred = fault_model.predict(X)
            prob = float(pred[0][0])
            pred_label = 1 if prob > 0.5 else 0
            msg = f"机床{machine_id}对话推理：{'故障' if pred_label==1 else '正常'} (概率 {prob:.2f})，模拟标签：{label}"
            latest_sim_predict[machine_id] = msg
            socketio.emit('log_message', {'message': f'机床{machine_id}对话推理完成，结果：{msg}'})
    except Exception as e:
        socketio.emit('log_message', {'message': f'对话自动推理异常: {e}'})
    user_input = data['message']
    signal = data.get('signal', None)

    if not user_input:
        emit('log_message', {'message': "输入不能为空，请重新输入。"})
        return

    if "退出" in user_input or "再见" in user_input:
        emit('ai_response', {'message': "好的，再见！"})
        speak_text("好的，再见！")
        # 清空对话历史
        global conversation_history
        conversation_history = []
        return

    # 如果有信号数据，先本地推理

    try:
        # 在新线程中处理AI响应，避免阻塞WebSocket
        thread = threading.Thread(target=process_ai_response, args=(user_input,))
        thread.start()
    except Exception as e:
        print("程序出错:", e)
        emit('log_message', {'message': f"程序出错: {e}"})
        emit('ai_response', {'message': "抱歉，我暂时无法连接服务器。"})
        speak_text("抱歉，我暂时无法连接服务器。")

def process_ai_response(user_input):
    ai_reply = get_llm_response(user_input)
    speak_text(ai_reply)


# ========== 故障预测 SocketIO 事件 ==========
@socketio.on('predict_fault')
def handle_predict_fault(data):
    """
    data: { 'signal': [float, ...] }  # 前端上传的时序信号数据
    """
    if fault_model is None or scaler is None:
        emit('fault_result', {'result': '模型未加载，无法预测'})
        return
    try:
        signal = data.get('signal', None)
        if signal is None or not isinstance(signal, list) or len(signal) < 200:
            emit('fault_result', {'result': '请上传长度不少于200的时序数据'})
            return
        # 归一化
        sig_arr = np.array(signal).reshape(-1, 1)
        sig_scaled = scaler.transform(sig_arr).flatten()
        # 取最后200个点作为一条序列
        seq = sig_scaled[-200:]
        X = seq.reshape(1, 200, 1)
        # 预测
        pred = fault_model.predict(X)
        prob = float(pred[0][0])
        label = 1 if prob > 0.5 else 0
        msg = f"预测结果：{'故障' if label==1 else '正常'} (概率 {prob:.2f})"
        emit('fault_result', {'result': msg, 'prob': prob, 'label': int(label)})
    except Exception as e:
        emit('fault_result', {'result': f'预测出错: {e}'})

# 添加清除历史记录的路由
@app.route('/clear_history', methods=['POST'])
def clear_history():
    global conversation_history
    conversation_history = []
    return jsonify({'status': 'success'})

@socketio.on('clear_history')
def handle_clear_history():
    global conversation_history
    conversation_history = []
    emit('history_cleared', {'message': '对话历史已清除'})

# ========== 主程序入口 ==========
if __name__ == "__main__":
    print("=== 数控机床AI语音助手已启动 ===")
    print("提示：访问 http://localhost:5000 使用网页版")
    # 初始化TTS事件循环
    init_tts_loop()
    socketio.run(app, debug=True, host='0.0.0.0')