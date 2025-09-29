# 阿里云百炼Paraformer语音识别API封装
import requests
import base64
import json

def ali_asr_recognize(audio_base64, apikey, appkey, format="wav", sample_rate=16000):
    """
    调用阿里云百炼Paraformer语音识别API。
    :param audio_base64: 录音的base64字符串（不含data:前缀）
    :param apikey: 阿里云百炼API Key
    :param appkey: 阿里云ASR服务appkey
    :param format: 音频格式，默认wav
    :param sample_rate: 采样率，默认16000
    :return: 识别文本
    """
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/asr/recognition"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {apikey}"
    }
    payload = {
        "model": "paraformer-16k-zh",  # 百炼Paraformer中文模型
        "input": {
            "audio": audio_base64,
            "format": format,
            "sample_rate": sample_rate
        },
        "parameters": {
            "appkey": appkey
        }
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        try:
            return data["output"]["text"]
        except Exception:
            return data.get("output", {}).get("text", "[解析ASR回复失败]")
    else:
        return f"[ASR API请求失败] {resp.status_code}: {resp.text}"
