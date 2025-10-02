# 通义千问Plus API 封装（优先使用 DashScope SDK 流式，以支持“硬中断”）
import requests
import json

def get_qwen_response(question, history=None, apikey=None, model="qwen-plus", cancel_event=None, on_update=None, streaming=True):
    """
    获取通义千问回复。
    - 优先使用 DashScope SDK 的流式输出，支持在生成过程中检查 cancel_event 并中断连接；
    - 若 SDK 或流式调用失败，回退到 REST 同步接口（此回退不支持中断）。

    参数：
    - question: str
    - history: list[{"role": "user"|"assistant", "content": str}]
    - apikey: str
    - model: str
    - cancel_event: threading.Event | None
    - on_update: Callable[[str], None] | None  # 可选：每次追加新文本时回调（传累计文本）
    - streaming: bool  默认 True，流式优先
    """
    if apikey is None:
        raise ValueError("请提供通义千问API Key")

    messages = list(history) if history else []
    messages.append({"role": "user", "content": question})

    if streaming:
        try:
            import dashscope  # 运行时已由主程序注入 api_key
            from dashscope import Generation
            dashscope.api_key = apikey

            # 开启流式与增量输出；result_format='message' 便于与 messages 格式对齐
            responses = Generation.call(
                model=model,
                messages=messages,
                stream=True,
                incremental_output=True,
                result_format='message'
            )
            acc = []
            # 逐事件读取，便于在 cancel_event.set() 后尽快中断
            for event in responses:
                # 硬中断：主动关闭底层流（如果支持），并抛出异常给上层识别
                if cancel_event is not None and getattr(cancel_event, 'is_set', lambda: False)():
                    try:
                        if hasattr(responses, 'close'):
                            responses.close()
                    except Exception:
                        pass
                    raise RuntimeError('CANCELLED')
                try:
                    # SDK 事件对象可能提供 output_text 或结构化 output
                    if hasattr(event, 'output_text') and event.output_text:
                        delta = event.output_text
                    elif hasattr(event, 'output') and event.output:
                        out = event.output
                        # 尝试不同字段获取文本
                        delta = (
                            out.get('text')
                            or out.get('choices', [{}])[0].get('message', {}).get('content')
                            or ''
                        )
                    else:
                        delta = ''
                except Exception:
                    delta = ''
                if delta:
                    acc.append(delta)
                    if on_update:
                        try:
                            on_update(''.join(acc))
                        except Exception:
                            pass
            return ''.join(acc) if acc else '[解析AI回复失败]'
        except Exception:
            # 流式失败时将回退到 REST
            pass

    # 回退：REST 接口（不支持中断）
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {apikey}"
    }
    payload = {
        "model": model,
        "input": {
            "messages": messages
        }
    }
    def _extract_text(data: dict) -> str:
        if not isinstance(data, dict):
            return ""
        # 优先从 output 中解析
        out = data.get("output")
        if isinstance(out, dict):
            # 直接 text 字段
            txt = out.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt
            # message.content
            msg = out.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    # 兼容多段内容
                    parts = []
                    for p in content:
                        if isinstance(p, dict):
                            parts.append(p.get("text") or p.get("content") or "")
                        elif isinstance(p, str):
                            parts.append(p)
                    joined = ''.join([x for x in parts if x])
                    if joined.strip():
                        return joined
            # choices[0].message.content / choices[0].content / choices[0].text
            chs = out.get("choices")
            if isinstance(chs, list) and chs:
                top = chs[0]
                if isinstance(top, dict):
                    cand = (
                        (isinstance(top.get("message"), dict) and top.get("message", {}).get("content"))
                        or top.get("content") or top.get("text")
                    )
                    if isinstance(cand, str) and cand.strip():
                        return cand
        # 顶层 choices 兜底
        chs = data.get("choices")
        if isinstance(chs, list) and chs:
            top = chs[0]
            if isinstance(top, dict):
                cand = (
                    (isinstance(top.get("message"), dict) and top.get("message", {}).get("content"))
                    or top.get("content") or top.get("text")
                )
                if isinstance(cand, str) and cand.strip():
                    return cand
        # 其它可能字段
        for k in ("text", "output_text", "result"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    except Exception as e:
        return f"[通义千问API请求失败] {e}"
    if resp.status_code == 200:
        try:
            data = resp.json()
        except Exception:
            return f"[通义千问API请求失败] 返回非JSON：{resp.text[:200]}"
        txt = _extract_text(data)
        return txt if txt else "[解析AI回复失败]"
    else:
        # 若返回体含错误码/错误信息，尽量抽取
        try:
            err = resp.json()
            code = err.get('code') or err.get('error_code') or resp.status_code
            msg = err.get('message') or err.get('error_msg') or resp.text
            return f"[通义千问API请求失败] {code}: {msg}"
        except Exception:
            return f"[通义千问API请求失败] {resp.status_code}: {resp.text}"
