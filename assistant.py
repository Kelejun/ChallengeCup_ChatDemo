import websocket
import hashlib
import base64
import hmac
import json
import time
import pyttsx3
from datetime import datetime, timezone
from urllib.parse import urlencode

# ========== 配置区：请务必确认你在讯飞平台获取的以下信息正确 ==========
APPID = "bf9978b7"           # 替换为你的真实 APPID（在控制台项目中查看）
APIKey = "d82430ad7a52eec5c133470dd68c5aec"     # 接口密钥中的 API Key
APISecret = "ZjZkNTIwNTZmNzYyYmYxMTgxY2U1YTMz"  # 接口密钥中的 API Secret

# ========== 初始化语音引擎 ==========
engine = pyttsx3.init()
engine.setProperty('rate', 180)      # 语速
engine.setProperty('volume', 0.9)    # 音量

# 全局变量用于接收响应
response_data = ""
received_response = False


def get_spark_response(question):
    global response_data, received_response
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

    # --- 准备发送的数据 ---
    prompt = f"""你是一个智能数控机床运维专家，请用简洁、专业、口语化的中文回答用户问题。
已知以下机床状态数据：
- 1号机床：主轴振动正常，温度72°C，处于稳定运行状态。
- 2号机床：轴承振动异常，预警等级为高，建议4小时内检修。
- 3号机床：一切正常。

用户问题：{question}
"""

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
                "text": [
                    {"role": "user", "content": prompt}
                ]
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
                response_data = "抱歉，服务端返回错误。"
                received_response = True
                ws.close()
                return

            # 获取回复内容
            content = msg["payload"]["choices"]["text"][0]["content"]
            response_data += content

            # 判断是否是最后一帧（status == 2 表示结束）
            if msg["header"]["status"] == 2:
                print("\n🔚 AI 回复接收完成。")
                received_response = True
                ws.close()

        except Exception as e:
            print("解析消息失败:", e)
            response_data = "解析响应失败。"
            received_response = True
            ws.close()

    def on_error(ws, error):
        print("WebSocket 错误:", error)
        ws.close()

    def on_close(ws, close_status_code, close_msg):
        print("WebSocket 连接已关闭")

    def on_open(ws):
        print("WebSocket 连接成功，正在发送问题...")
        ws.send(json.dumps(data))

    # --- 建立 WebSocket 连接 ---
    print("正在连接星火大模型 API...")
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
    return response_data


# ========== 文本转语音函数 ==========
def speak_text(text):
    if text and text.strip():
        print(f"AI正在说: {text}")
        engine.say(text)
        engine.runAndWait()
    else:
        print("没有内容可朗读")


# ========== 主程序入口 ==========
if __name__ == "__main__":
    print("=== 数控机床AI语音助手已启动 ===")
    print("提示：输入‘退出’可结束程序\n")

    while True:
        user_input = input("请输入你的问题 > ").strip()

        if not user_input:
            print("输入不能为空，请重新输入。")
            continue

        if "退出" in user_input or "再见" in user_input:
            speak_text("好的，再见！")
            break

        try:
            ai_reply = get_spark_response(user_input)
            speak_text(ai_reply)
        except Exception as e:
            print("程序出错:", e)
            speak_text("抱歉，我暂时无法连接服务器。")

    print("程序已退出，欢迎下次使用！")