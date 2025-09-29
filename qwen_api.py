# 通义千问Plus API 封装
import requests
import json

def get_qwen_response(question, history=None, apikey=None, model="qwen-plus"):
    """
    调用通义千问Plus API获取回复。
    :param question: 用户问题字符串
    :param history: 可选，对话历史，格式为[{"role": "user"/"assistant", "content": "..."}]
    :param apikey: 通义千问API Key
    :param model: 模型名称，默认qwen-plus
    :return: AI回复字符串
    """
    if apikey is None:
        raise ValueError("请提供通义千问API Key")
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {apikey}"
    }
    messages = history if history else []
    messages.append({"role": "user", "content": question})
    payload = {
        "model": model,
        "input": {
            "messages": messages
        }
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        try:
            return data["output"]["text"]
        except Exception:
            return data.get("output", {}).get("text", "[解析AI回复失败]")
    else:
        return f"[通义千问API请求失败] {resp.status_code}: {resp.text}"
