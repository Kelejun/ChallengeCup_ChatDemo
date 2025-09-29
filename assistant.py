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

# ========== 配置区：请务必确认你在讯飞平台获取的以下信息正确 ==========
APPID = "bf9978b7"           # 替换为你的真实 APPID（在控制台项目中查看）
APIKey = "d82430ad7a52eec5c133470dd68c5aec"     # 接口密钥中的 API Key
APISecret = "ZjZkNTIwNTZmNzYyYmYxMTgxY2U1YTMz"  # 接口密钥中的 API Secret

# ========== 初始化语音引擎 ==========
# 使用线程锁来确保TTS引擎的线程安全
tts_lock = threading.Lock()

def get_tts_engine():
    """获取TTS引擎实例，每次创建新实例避免冲突"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)      # 语速
    engine.setProperty('volume', 0.9)    # 音量
    return engine

# 全局变量用于接收响应
response_data = ""
received_response = False

# ========== 存储对话历史 ==========
conversation_history = []

# ========== Flask Web 应用 ==========
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

def get_spark_response(question):
    global response_data, received_response, conversation_history
    response_data = ""
    received_response = False

    # --- 请求配置 ---
    host = "spark-api.xf-yun.com"
    path = "/v1.1/chat"
    url = f"wss://{host}{path}"

    # --- 当前 UTC 时间（GMT 格式）---
    now = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')

    # --- 构造 signature_origin 字符串（注意换行和空格）---
    signature_origin = f"host: {host}\ndate: {now}\nGET {path} HTTP/1.1"

    # --- 计算 HMAC-SHA256 签名 ---
    signature_sha = hmac.new(
        APISecret.encode('utf-8'),
        signature_origin.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    signature_b64 = base64.b64encode(signature_sha).decode('utf-8')

    # --- 构造 authorization 头字段（字符串形式）---
    authorization_str = f'api_key="{APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_b64}"'

    # --- 将 authorization 字符串 Base64 编码作为 URL 参数 ---
    authorization_enc = base64.b64encode(authorization_str.encode('utf-8')).decode('utf-8')

    # --- 构造最终请求 URL ---
    params = {
        "host": host,
        "date": now,
        "authorization": authorization_enc
    }
    request_url = f"{url}?{urlencode(params)}"
    print("请求URL已生成")
    socketio.emit('log_message', {'message': '请求URL已生成'})

    # --- 准备发送的数据 ---
    prompt = f"""你是一个智能数控机床运维专家，请用简洁、专业、口语化的中文回答用户问题。
已知以下机床状态数据：
- 1号机床：主轴振动正常，温度72°C，处于稳定运行状态。
- 2号机床：轴承振动异常，预警等级为高，建议4小时内检修。
- 3号机床：一切正常。

用户问题：{question}
"""

    # 添加当前问题到对话历史
    conversation_history.append({"role": "user", "content": prompt})
    
    # 限制对话历史长度，防止超出token限制
    if len(conversation_history) > 10:  # 保留最近5轮对话
        conversation_history = conversation_history[-10:]

    data = {
        "header": {"app_id": APPID},
        "parameter": {
            "chat": {
                "domain": "lite",
                "temperature": 0.5,
                "max_tokens": 2048
            }
        },
        "payload": {
            "message": {
                "text": conversation_history
            }
        }
    }

    # --- WebSocket 回调函数 ---
    def on_message(ws, message):
        global response_data, received_response
        try:
            msg = json.loads(message)
            code = msg["header"]["code"]
            if code != 0:
                print(f"API 返回错误码 {code}: {msg['header']['message']}")
                socketio.emit('log_message', {'message': f"API 返回错误码 {code}: {msg['header']['message']}"})
                response_data = "抱歉，服务端返回错误。"
                socketio.emit('ai_response', {'message': response_data})
                received_response = True
                ws.close()
                return

            # 获取回复内容
            content = msg["payload"]["choices"]["text"][0]["content"]
            response_data += content
            socketio.emit('ai_response_update', {'message': content})

            # 判断是否是最后一帧（status == 2 表示结束）
            if msg["header"]["status"] == 2:
                print("\n🔚 AI 回复接收完成。")
                socketio.emit('log_message', {'message': 'AI 回复接收完成'})
                # 将AI回复添加到对话历史
                conversation_history.append({"role": "assistant", "content": response_data})
                received_response = True
                ws.close()

        except Exception as e:
            print("解析消息失败:", e)
            socketio.emit('log_message', {'message': f"解析消息失败: {e}"})
            response_data = "解析响应失败。"
            socketio.emit('ai_response', {'message': response_data})
            received_response = True
            ws.close()

    def on_error(ws, error):
        print("WebSocket 错误:", error)
        socketio.emit('log_message', {'message': f"WebSocket 错误: {error}"})
        ws.close()

    def on_close(ws, close_status_code, close_msg):
        print("WebSocket 连接已关闭")
        socketio.emit('log_message', {'message': "WebSocket 连接已关闭"})

    def on_open(ws):
        print("WebSocket 连接成功，正在发送问题...")
        socketio.emit('log_message', {'message': 'WebSocket 连接成功，正在发送问题...'})
        ws.send(json.dumps(data))

    # --- 建立 WebSocket 连接 ---
    print("正在连接星火大模型 API...")
    socketio.emit('log_message', {'message': '正在连接星火大模型 API...'})
    ws = websocket.WebSocketApp(
        request_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever(ping_interval=6, ping_timeout=3)  # 自动保活

    # --- 等待响应完成 ---
    while not received_response:
        time.sleep(0.1)

    print(f"AI回答: {response_data}")
    socketio.emit('ai_response', {'message': response_data})
    return response_data


# ========== 文本转语音函数 ==========
def speak_text(text):
    if text and text.strip():
        print(f"AI正在说: {text}")
        socketio.emit('log_message', {'message': f"AI正在说: {text}"})
        
        def _speak():
            try:
                # 在独立线程中创建新的TTS引擎实例，避免事件循环冲突
                with tts_lock:  # 使用锁确保线程安全
                    engine = get_tts_engine()
                    engine.say(text)
                    engine.runAndWait()
                    # 主动停止引擎释放资源
                    engine.stop()
                print("TTS播放完成")
                socketio.emit('log_message', {'message': "TTS播放完成"})
            except Exception as e:
                print(f"TTS播放出错: {e}")
                socketio.emit('log_message', {'message': f"TTS播放出错: {e}"})
        
        # 简化TTS执行逻辑，直接在新线程中播放
        # 避免复杂的异步事件循环嵌套
        try:
            tts_thread = threading.Thread(target=_speak, daemon=True)
            tts_thread.start()
        except Exception as e:
            print(f"创建TTS线程失败: {e}")
            socketio.emit('log_message', {'message': f"创建TTS线程失败: {e}"})
            # 如果线程创建失败，直接在当前线程执行
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
    user_input = data['message']
    
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
    ai_reply = get_spark_response(user_input)
    speak_text(ai_reply)

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
    try:
        socketio.run(app, debug=True, host='0.0.0.0')
    except SystemExit:
        # 正常退出，不处理
        pass
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
