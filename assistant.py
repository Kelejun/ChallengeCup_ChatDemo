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
import os
from typing import Optional
from multiprocessing import Process, Queue, freeze_support
from multiprocessing.queues import Queue as MPQueue
import dashscope

# ========== 配置区：通义千问Plus & 阿里云ASR API Key ==========
QWEN_API_KEY = "sk-633484a178eb422d9f1969320be1f25f"  # 通义千问API Key（与阿里云百炼ASR共用）
ALI_ASR_API_KEY = QWEN_API_KEY  # 阿里云百炼API Key自动同步千问APIKey

# 统一设置DashScope API Key（SDK建议使用环境变量/全局变量任一方式）
os.environ['DASHSCOPE_API_KEY'] = ALI_ASR_API_KEY
dashscope.api_key = ALI_ASR_API_KEY

# ========== 提前初始化Flask和SocketIO，确保后续所有@socketio.on可用 ==========
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ========== 日志持久化：写入 log/YYYY-MM-DD.log，并广播到前端 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'log')

def log(msg: str):
    """将日志写入按日期分文件的 log 目录，并通过 SocketIO 广播到前端。"""
    try:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR, exist_ok=True)
        now = datetime.now()
        ts = now.strftime('%Y-%m-%d %H:%M:%S')
        file_path = os.path.join(LOG_DIR, f"{now.strftime('%Y-%m-%d')}.log")
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception as e:
        # 文件写入失败不影响运行，打印到控制台
        print(f"[LOG WRITE FAIL] {e}: {msg}")
    try:
        socketio.emit('log_message', {'message': msg})
    except Exception as e:
        print(f"[LOG EMIT FAIL] {e}: {msg}")

# ========== 故障预测模型相关 ========== 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
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
            result_text = f"{'故障' if pred_label==1 else '正常'} (概率 {prob:.2f})，模拟标签：{label}"
            latest_sim_predict[machine_id] = f"机床{machine_id}初始推理：{result_text}"
            print(f'机床{machine_id}初始推理完成，结果：{result_text}')
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

# ========== 导入通义千问API和ASR封装 ========== 
from qwen_api import get_qwen_response
from ali_asr import ali_asr_recognize

# ========== 语音识别SocketIO事件 ==========
@socketio.on('asr_audio')
def handle_asr_audio(data):
    """
    data: { audio: base64字符串（不含data:前缀） }
    """
    audio_base64 = data.get('audio', None)
    if not audio_base64:
        log('[ASR] 未收到音频数据')
        emit('asr_result', {'text': '', 'error': '未收到音频数据'})
        return
    log(f'[ASR] 收到音频数据，长度: {len(audio_base64)} 字节')
    if not QWEN_API_KEY:
        log('[ASR] ASR配置缺失，请检查API Key')
        emit('asr_result', {'text': '', 'error': 'ASR配置缺失'})
        return
    try:
        asr_model = "paraformer-realtime-8k-v2"
        log(f'[ASR] 开始识别，模型: {asr_model}，采样率: 8000Hz，目标格式: wav')
        result = ali_asr_recognize(audio_base64, apikey=ALI_ASR_API_KEY, format="wav", sample_rate=8000)

        # 结构化结果处理
        if isinstance(result, dict):
            if result.get('ok'):
                text = result.get('text', '')
                log(f'[ASR] 识别成功: {text}')
                emit('asr_result', {'text': text})
            else:
                code = result.get('code', 'UNKNOWN')
                detail = result.get('detail')
                # 统一错误提示文案
                friendly = {
                    'SDK_IMPORT_FAIL': 'ASR SDK 未安装或导入失败，请安装 dashscope 包并重启',
                    'INPUT_UNSUPPORTED': '音频格式不支持，请使用 PCM WAV 8k 单声道',
                    'SDK_NO_OUTPUT': '识别完成但未返回文本',
                    'EXCEPTION': 'ASR 识别异常',
                    'RESULT_SERIALIZE_ERROR': 'ASR 结果解析失败'
                }.get(code, 'ASR 未知错误')

                log(f'[ASR] {friendly} (code={code})')
                # 追加诊断信息到日志（不回传给输入框）
                if detail is not None:
                    try:
                        log(f"[ASR] 诊断: {json.dumps(detail, ensure_ascii=False)}")
                    except Exception:
                        log(f"[ASR] 诊断: {detail}")
                emit('asr_result', {'text': '', 'error': friendly, 'code': code})
        else:
            # 兼容老返回（字符串）
            text = str(result) if result is not None else ''
            if text:
                log(f'[ASR] 识别成功: {text}')
                emit('asr_result', {'text': text})
            else:
                log('[ASR] 识别完成，但未返回文本')
                emit('asr_result', {'text': '', 'error': '识别完成但未返回文本'})
    except Exception as e:
        log(f'[ASR] 识别异常: {e}')
        emit('asr_result', {'text': '', 'error': str(e)})

# ========== 导入通义千问API封装 ==========
from qwen_api import get_qwen_response


# ========== 语音参数 ========== 
TTS_RATE = 180
TTS_VOLUME = 0.9

# 全局变量用于接收响应
response_data = ""
received_response = False


conversation_history = []
# 存储最近一次模拟推送的预测结果
latest_sim_predict = {}  # {machine_id: 推理结果字符串}

# ========== 后台健康检查（每5分钟） ==========
_health_last_state = {
    # machine_id: bool -> 上次是否为故障（True）
}
_health_model_alerted = False  # 模型未就绪是否已告警

def _health_check_once():
    global _health_last_state, _health_model_alerted
    try:
        # 巡检开始日志（无论是否有异常，都记录一次开始）
        log('[健康检查] 开始巡检')
        anomalies = []  # 收集本轮所有异常，用于合并通知AI
        if fault_model is None or scaler is None:
            if not _health_model_alerted:
                msg = '[健康检查] 模型或归一化器未加载，无法执行巡检'
                log(msg)
                _payload = {
                    'level': 'error',
                    'code': 'MODEL_NOT_READY',
                    'message': '模型未加载，自动健康检查不可用，请检查服务配置',
                    'ts': time.time()
                }
                socketio.emit('system_alert', _payload)
                try:
                    log(f"[系统警报][payload] {json.dumps(_payload, ensure_ascii=False)}")
                except Exception:
                    log(f"[系统警报][payload] {_payload}")
                _health_model_alerted = True
                anomalies.append({
                    'type': 'MODEL_NOT_READY',
                    'source': '系统',
                    'message': '模型或归一化器未加载',
                })
            # 即便模型未就绪，也可以将该异常作为一次AI通知（单条/合并均可）
            if anomalies:
                try:
                    threading.Thread(target=_notify_ai_for_anomalies, args=(anomalies,), daemon=True).start()
                except Exception:
                    pass
            # 巡检结束日志（模型未就绪）
            log('[健康检查] 巡检结束（模型未就绪）')
            return

        import scipy.io, random
        normal = scipy.io.loadmat('97.mat')["X097_DE_time"].flatten()
        fault = scipy.io.loadmat('105.mat')["X105_DE_time"].flatten()
        for machine_id in range(1, 4):
            # 模拟从正常/故障源中抽取一段作为“当前窗口”，用模型判断
            label = random.choice(['normal', 'fault'])
            arr = normal if label == 'normal' else fault
            if len(arr) < 200:
                continue
            start = random.randint(0, len(arr) - 200)
            segment = arr[start:start+200]
            sig_arr = np.array(segment).reshape(-1, 1)
            sig_scaled = scaler.transform(sig_arr).flatten()
            seq = sig_scaled[-200:]
            X = seq.reshape(1, 200, 1)
            pred = fault_model.predict(X)
            prob = float(pred[0][0])
            is_fault = prob > 0.5

            # 去重复告警：仅在状态从非故障 -> 故障 时告警
            prev = _health_last_state.get(machine_id, False)
            if is_fault and not prev:
                alert_msg = f'警报：机床{machine_id} 检测到可能故障 (概率 {prob:.2f})，请及时检修'
                log(f'[健康检查] {alert_msg}')
                _payload = {
                    'level': 'warning',
                    'code': 'PREDICTED_FAULT',
                    'machine_id': machine_id,
                    'prob': prob,
                    'message': alert_msg,
                    'ts': time.time()
                }
                socketio.emit('system_alert', _payload)
                try:
                    log(f"[系统警报][payload] {json.dumps(_payload, ensure_ascii=False)}")
                except Exception:
                    log(f"[系统警报][payload] {_payload}")
                anomalies.append({
                    'type': 'PREDICTED_FAULT',
                    'source': '健康检查-预测故障',
                    'machine_id': machine_id,
                    'prob': prob,
                    'message': alert_msg,
                })
            # 更新状态；正常不告警且静默
            _health_last_state[machine_id] = is_fault
        # 合并AI通知（若有1条以上，合并发送；1条也支持复用该方式）
        if anomalies:
            try:
                threading.Thread(target=_notify_ai_for_anomalies, args=(anomalies,), daemon=True).start()
                log(f'[健康检查] 已合并{len(anomalies)}条异常提交AI建议')
            except Exception:
                pass
            # 巡检结束日志（发现异常）
            log(f'[健康检查] 巡检结束（发现 {len(anomalies)} 条异常）')
        else:
            # 巡检结束日志（无异常）
            log('[健康检查] 巡检结束（无异常）')
    except Exception as e:
        log(f'[健康检查] 异常: {e}')

def _health_check_loop():
    # 首次延迟2分钟，给模型加载与服务稳定留时间；之后每5分钟巡检一次
    try:
        time.sleep(120)
    except Exception:
        pass
    while True:
        _health_check_once()
        try:
            time.sleep(300)  # 5分钟
        except Exception:
            pass

def _notify_ai_for_anomalies(anomalies: list):
    """后台线程：合并异常，向AI提交一次问题，获取建议并播报。"""
    try:
        if not QWEN_API_KEY:
            log('[健康检查] AI通知未发送：缺少通义千问API Key')
            return
        # 构造简要多异常摘要
        lines = []
        for a in anomalies:
            t = a.get('type', 'UNKNOWN')
            src = a.get('source', '系统')
            if t == 'PREDICTED_FAULT':
                mid = a.get('machine_id')
                prob = a.get('prob', 0.0)
                lines.append(f"- [来源:{src}] 机床{mid} 故障概率 {prob:.2f}")
            elif t == 'MODEL_NOT_READY':
                lines.append(f"- [来源:{src}] 模型未就绪，无法执行健康检查")
            else:
                msg = a.get('message', '未知异常')
                lines.append(f"- [来源:{src}] {msg}")
        summary = "\n".join(lines)
        plural = '多条异常' if len(anomalies) > 1 else '异常'
        log(f'[健康检查] 已向AI提交{plural}，请求应对建议...')
        question = (
            "系统健康检查检测到以下异常：\n" + summary + "\n"
            "请以机床运维专家的角度，用简洁中文给出风险评估（低/中/高），"
            "并按优先级给出2-3条具体行动建议（如检查润滑、采集更多振动数据、安排停机检测等），"
            "每条尽量不超过20字，总字数不超过80字，整段话应该都是口语化的，以“警告：”开头。"
        )
        reply = get_llm_response(question)
        speak_text(reply)
    except Exception as e:
        log(f'[健康检查] 向AI发送异常通知失败: {e}')

# ========== 接收模拟信号并自动预测 ==========
@socketio.on('simulated_signal')
def handle_simulated_signal(data):
    """
    data: { 'signal': [...], 'label': 'normal'|'fault' }
    """
    global latest_sim_predict
    if fault_model is None or scaler is None:
        log('[推理] 模型或归一化器未加载，无法预测')
        emit('fault_result', {'result': '模型未加载，无法预测'})
        return
    try:
        log('[推理] 开始处理模拟信号...')
        signal = data.get('signal', None)
        label = data.get('label', '')
        if signal is None or not isinstance(signal, list) or len(signal) < 200:
            log('[推理] 信号数据无效或长度不足200，跳过')
            emit('fault_result', {'result': '模拟数据长度不足200'})
            return
        log(f'[推理] 信号数据加载完成，长度: {len(signal)}')
        sig_arr = np.array(signal).reshape(-1, 1)
        log('[推理] 数据归一化中...')
        sig_scaled = scaler.transform(sig_arr).flatten()
        seq = sig_scaled[-200:]
        X = seq.reshape(1, 200, 1)
        log('[推理] 开始模型推理...')
        pred = fault_model.predict(X)
        prob = float(pred[0][0])
        pred_label = 1 if prob > 0.5 else 0
        msg = f"模拟推送数据预测：{'故障' if pred_label==1 else '正常'} (概率 {prob:.2f})，真实标签：{label}"
        latest_sim_predict[0] = msg
        log(f'[推理] 推理完成，结果：{msg}')
        emit('sim_predict_result', {'result': msg, 'prob': prob, 'label': int(pred_label), 'true_label': label})
    except Exception as e:
        log(f'[推理] 推理异常: {e}')
        emit('sim_predict_result', {'result': f'模拟数据预测出错: {e}'})


# ========== TTS 多进程 ==========
tts_queue: Optional[MPQueue] = None
tts_process: Optional[Process] = None

def tts_process_main(q: Queue, rate: int, volume: float):
    # 子进程循环：每条文本都新建一次引擎，避免状态残留导致后续不发声
    while True:
        try:
            text = q.get()
        except Exception:
            continue
        if text is None:
            break
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', rate)
            engine.setProperty('volume', volume)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS朗读异常: {e}")
        finally:
            try:
                # 显式销毁引用，帮助释放底层资源
                del engine
            except Exception:
                pass

def start_tts_process():
    global tts_queue, tts_process
    if tts_process and tts_process.is_alive():
        return
    tts_queue = Queue()
    tts_process = Process(target=tts_process_main, args=(tts_queue, TTS_RATE, TTS_VOLUME), daemon=True)
    tts_process.start()

def stop_tts_process():
    global tts_queue, tts_process
    try:
        if tts_queue:
            tts_queue.put(None)
    except Exception:
        pass
    if tts_process and tts_process.is_alive():
        tts_process.join(timeout=1)


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
请用*简洁、专业、尽量口语化*的中文回答用户问题，并给出必要的建议，如及时安排检修等。最后，不要使用Markdown格式。{sim_info}
\n用户问题：{question}
"""
    conversation_history.append({"role": "user", "content": prompt})
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    try:
        log('[API] 正在连接通义千问API...')
        reply = get_qwen_response(question=prompt, history=conversation_history[:-1], apikey=QWEN_API_KEY)
        log('[API] API响应成功')
        conversation_history.append({"role": "assistant", "content": reply})
        socketio.emit('ai_response', {'message': reply})
        try:
            log(f"AI: {reply}")
        except Exception:
            pass
        return reply
    except Exception as e:
        err = f"[通义千问API调用失败] {e}"
        log(f'[API] API调用异常: {e}')
        socketio.emit('ai_response', {'message': err})
        try:
            log(f"AI: {err}")
        except Exception:
            pass
        return err


# ========== 文本转语音函数 ==========
def speak_text(text):
    if text and text.strip():
        print(f"AI正在说: {text}")
        log(f"AI正在说: {text}")
        # 将文本发送到TTS进程；若进程未启动则尝试降级为本进程朗读
        try:
            # 若TTS进程异常退出，尝试自动重启
            global tts_process
            if tts_process is None or (tts_process is not None and not tts_process.is_alive()):
                start_tts_process()
                log('[TTS] 子进程已重启')
            if tts_queue is not None:
                tts_queue.put(text)
            else:
                # 降级：本进程直接朗读（可能阻塞，尽量避免）
                engine = pyttsx3.init()
                engine.setProperty('rate', TTS_RATE)
                engine.setProperty('volume', TTS_VOLUME)
                engine.say(text)
                engine.runAndWait()
        except Exception as e:
            print(f"TTS队列发送失败，降级播放异常: {e}")
    else:
        print("没有内容可朗读")
        log("没有内容可朗读")


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
            result_text = f"{'故障' if pred_label==1 else '正常'} (概率 {prob:.2f})，模拟标签：{label}"
            latest_sim_predict[machine_id] = f"机床{machine_id}对话推理：{result_text}"
            log(f'[推理] 机床{machine_id}对话推理完成，结果：{result_text}')
    except Exception as e:
        log(f'对话自动推理异常: {e}')
    user_input = data['message']
    signal = data.get('signal', None)

    if not user_input:
        log("输入不能为空，请重新输入。")
        return
    # 将用户输入持久化到日志
    try:
        log(f"用户: {user_input}")
    except Exception:
        pass

    if "退出" in user_input or "再见" in user_input:
        emit('ai_response', {'message': "好的，再见！"})
        try:
            log("AI: 好的，再见！")
        except Exception:
            pass
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
        log(f"程序出错: {e}")
        emit('ai_response', {'message': "抱歉，我暂时无法连接服务器。"})
        try:
            log("AI: 抱歉，我暂时无法连接服务器。")
        except Exception:
            pass
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

# ========== 历史日志加载（跨天连续，基于复合游标 {date, offset} 向前分页） ==========
@socketio.on('load_history')
def handle_load_history(data):
    """
    请求参数：
      - cursor: 可选，形如 { 'date': 'YYYY-MM-DD', 'offset': <int> }，表示该日期从文件末尾已读取 offset 行
      - limit:  本次最多返回的记录条数，默认 100

    行为：
      - 若未提供 cursor，则自动从最新一天的日志开始向前查找
      - 会跨越多天，直到凑满 limit 条或没有更早的日志

    返回：
      - items: [{ts, kind: 'user'|'ai'|'log', text}]
      - next_cursor: 若还有更多历史，则给出下一次查询用的 {date, offset}；否则为 None
      - has_more: 是否还有更多历史
      - date: 为兼容前端旧字段，返回 next_cursor 的 date 或当前起始日期
    """
    try:
        limit = int((data or {}).get('limit') or 100)
        cursor_in = (data or {}).get('cursor')
        try:
            log(f"[历史] 请求: cursor={cursor_in}, limit={limit}")
        except Exception:
            pass

        # 列出所有可用日志日期（YYYY-MM-DD.log），升序排列
        if not os.path.exists(LOG_DIR):
            emit('history_chunk', {'items': [], 'next_cursor': None, 'has_more': False, 'date': None})
            return
        files = [f for f in os.listdir(LOG_DIR) if f.endswith('.log')]
        dates = []
        for f in files:
            name = f[:-4]
            try:
                # 验证日期格式
                datetime.strptime(name, '%Y-%m-%d')
                dates.append(name)
            except Exception:
                continue
        dates.sort()  # 升序
        try:
            log(f"[历史] 可用日期: {dates}")
        except Exception:
            pass
        if not dates:
            emit('history_chunk', {'items': [], 'next_cursor': None, 'has_more': False, 'date': None})
            return

        # 确定起始日期与偏移
        if cursor_in and isinstance(cursor_in, dict) and cursor_in.get('date') in dates:
            curr_date = cursor_in.get('date')
            curr_offset = int(cursor_in.get('offset') or 0)
        else:
            # 默认从最新日期开始
            curr_date = dates[-1]
            curr_offset = 0

        curr_index = dates.index(curr_date)
        remaining = limit
        batch_items = []  # 按整体时间从早到晚排序
        ended_date = curr_date
        ended_offset = curr_offset

        def parse_line(line: str):
            ts = ''
            msg = line
            if line.startswith('['):
                try:
                    idx = line.index(']')
                    ts = line[1:idx].strip()
                    msg = line[idx+1:].lstrip()
                except Exception:
                    ts = ''
                    msg = line
            kind = 'log'
            text = msg
            if msg.startswith('用户: '):
                kind = 'user'
                text = msg[4:].strip()
            elif msg.startswith('AI: '):
                kind = 'ai'
                text = msg[3:].strip()
            return {'ts': ts, 'kind': kind, 'text': text}

        while remaining > 0 and curr_index >= 0:
            file_path = os.path.join(LOG_DIR, f"{dates[curr_index]}.log")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
            except Exception:
                # 文件读取异常则跳过该天
                curr_index -= 1
                curr_offset = 0
                continue

            total = len(lines)
            available = max(0, total - curr_offset)
            if available <= 0:
                curr_index -= 1
                curr_offset = 0
                continue

            read_count = min(remaining, available)
            start = max(0, total - curr_offset - read_count)
            end = max(0, total - curr_offset)
            chunk = lines[start:end]
            parsed_chunk = [parse_line(l) for l in chunk]  # 该子块内为按时间从早到晚
            # 为满足“倒序（最新在前）”需求：将子块反转为从新到旧
            parsed_chunk_rev = list(reversed(parsed_chunk))
            try:
                log(f"[历史] 读取: {dates[curr_index]} 行 {start}:{end}（返回 {len(parsed_chunk_rev)} 条，倒序）")
            except Exception:
                pass

            # 整体批次维持从新到旧（按日期从近到远），因此将当前天的倒序子块追加到末尾
            batch_items = batch_items + parsed_chunk_rev

            curr_offset += read_count
            ended_date = dates[curr_index]
            ended_offset = curr_offset
            remaining -= read_count

            if remaining > 0:
                # 继续走到更早一天
                curr_index -= 1
                curr_offset = 0

        # 判断是否还有更多历史，并给出下一游标
        has_more = False
        next_cursor = None
        if remaining == 0:
            # 本次已凑满 limit 条
            try:
                with open(os.path.join(LOG_DIR, f"{ended_date}.log"), 'r', encoding='utf-8') as f:
                    total_ended = len(f.read().splitlines())
            except Exception:
                total_ended = 0
            ended_idx = dates.index(ended_date) if ended_date in dates else -1
            if total_ended - ended_offset > 0:
                # 同一天仍有未读
                has_more = True
                next_cursor = { 'date': ended_date, 'offset': ended_offset }
            elif ended_idx - 1 >= 0:
                # 当天已读尽，但仍有更早日期，直接跳转到前一天起点
                has_more = True
                prev_date = dates[ended_idx - 1]
                next_cursor = { 'date': prev_date, 'offset': 0 }
        else:
            # 未凑满，说明要么越界（没有更早天），要么已切换到更早一天但尚未读取
            if curr_index >= 0:
                has_more = True
                next_cursor = { 'date': dates[curr_index], 'offset': 0 }
            else:
                has_more = False
                next_cursor = None

        try:
            log(f"[历史] 返回: items={len(batch_items)}, next_cursor={next_cursor}, has_more={has_more}")
        except Exception:
            pass
        emit('history_chunk', {
            'items': batch_items,
            'next_cursor': next_cursor,
            'has_more': has_more,
            'date': (next_cursor['date'] if isinstance(next_cursor, dict) else ended_date)
        })
    except Exception as e:
        emit('history_chunk', {
            'items': [],
            'error': str(e),
            'next_cursor': (data.get('cursor') if isinstance(data, dict) else None),
            'has_more': False,
            'date': (data.get('cursor', {}).get('date') if isinstance(data, dict) else None)
        })

# ========== 主程序入口 ==========
if __name__ == "__main__":
    freeze_support()  # Windows/PyInstaller 兼容
    print("=== 数控机床AI语音助手已启动 ===")
    print("提示：访问 http://localhost:5000 使用网页版")
    debug_flag = True
    # 服务启动日志
    try:
        log('[服务] 启动')
    except Exception:
        pass
    # 禁用自动重载器，确保仅运行一次，避免依赖 WERKZEUG_RUN_MAIN
    start_tts_process()
    # 启动后台健康检查线程（守护线程）
    try:
        t = threading.Thread(target=_health_check_loop, daemon=True)
        t.start()
        log('[健康检查] 后台巡检线程已启动')
    except Exception as e:
        log(f'[健康检查] 启动失败: {e}')
    try:
        socketio.run(app, debug=debug_flag, host='0.0.0.0', use_reloader=False)
    finally:
        try:
            log('[服务] 停止')
        except Exception:
            pass
        stop_tts_process()
