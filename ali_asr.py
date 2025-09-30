# 阿里云百炼Paraformer语音识别API封装

import websocket
import json

def ali_asr_recognize(audio_base64, apikey, format="wav", sample_rate=8000):
    """
    使用WebSocket调用阿里云百炼Paraformer实时语音识别API。
    :param audio_base64: 录音的base64字符串（不含data:前缀）
    :param apikey: 阿里云百炼API Key
    :param format: 音频格式，默认wav
    :param sample_rate: 采样率，默认8000
    :return: 识别文本或错误信息
    """
    ws_url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
    model_name = "paraformer-realtime-8k-v2"
    payload = {
        "model": model_name,
        "input": {
            "audio": audio_base64,
            "format": format,
            "sample_rate": sample_rate
        }
    }
    try:
        ws = websocket.create_connection(ws_url, header=[f"Authorization: Bearer {apikey}"])
        ws.send(json.dumps(payload))
        result = ""
        while True:
            resp = ws.recv()
            if not resp:
                break
            data = json.loads(resp)
            # 识别结果通常在output.text字段
            if "output" in data and "text" in data["output"]:
                result = data["output"]["text"]
                break
            # 错误信息
            if "code" in data and "message" in data:
                result = f"[ASR WS错误] {data['code']}: {data['message']}"
                break
        ws.close()
        return result
    except Exception as e:
        return f"[ASR WS异常] {e}"
    except Exception as e:
        return f"[ASR WS异常] {e}"
