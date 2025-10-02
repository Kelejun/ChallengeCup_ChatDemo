# 阿里云百炼Paraformer语音识别API封装（DashScope Python SDK 版）

import base64
import os
import tempfile
import dashscope
import subprocess
import shutil
try:
    from dashscope.audio.asr import Recognition
except ImportError:
    Recognition = None

def ali_asr_recognize(audio_base64, apikey, format="wav", sample_rate=8000):
    """
    使用阿里云DashScope官方Python SDK（Recognition类，同步调用）进行识别。

    输入：
    - audio_base64: 前端传来的不带 data: 前缀的 Base64 音频（推荐WAV/PCM/8k/mono）
    - apikey: DashScope API Key
    - format: 客户端声明的原始格式（用于命名临时文件后缀，实际已做嗅探）
    - sample_rate: 采样率（默认 8000）

    返回统一结构：
    {
        'ok': bool,
        'text': str,
        'code': str,            # ASR_OK / SDK_IMPORT_FAIL / INPUT_UNSUPPORTED / SDK_NO_OUTPUT / EXCEPTION / RESULT_SERIALIZE_ERROR
        'detail': dict|str|None
    }
    """
    if Recognition is None:
        return {
            'ok': False,
            'text': '',
            'code': 'SDK_IMPORT_FAIL',
            'detail': 'dashscope SDK 未导入成功，请安装 dashscope 并重启进程'
        }

    # 预初始化，避免 finally 中引用未定义
    tmp_path = None
    tmp_wav_path = None
    call_format = format or 'wav'
    has_ffmpeg = bool(shutil.which('ffmpeg'))
    # 预存原始 API Key 设置，确保 finally 能复原
    prev_key_env = os.environ.get('DASHSCOPE_API_KEY')
    prev_key_attr = getattr(dashscope, 'api_key', None)

    try:
        # 写入临时音频文件
        audio_bytes = base64.b64decode(audio_base64)
        suffix = f".{format.lower()}" if format else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        # 简单格式嗅探
        header = audio_bytes[:4]
        detected = None
        if header.startswith(b'RIFF'):
            detected = 'wav'
        elif header.startswith(b'OggS'):
            detected = 'opus'
        elif header == b'\x1aE\xDf\xa3' or header == b'\x1a\x45\xdf\xa3':
            detected = 'webm'

        # 若系统安装了ffmpeg，则统一转码为8kHz、单声道、PCM WAV
        call_path = tmp_path
        call_format = (detected or format or 'wav')
        if has_ffmpeg:
            try:
                tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                tmp_wav_path = tmp_wav.name
                tmp_wav.close()
                cmd = [
                    'ffmpeg', '-y', '-i', tmp_path,
                    '-ac', '1', '-ar', str(sample_rate),
                    '-acodec', 'pcm_s16le', tmp_wav_path
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                call_path = tmp_wav_path
                call_format = 'wav'
            except Exception:
                pass
        else:
            if detected == 'webm':
                return {
                    'ok': False,
                    'text': '',
                    'code': 'INPUT_UNSUPPORTED',
                    'detail': '检测到WebM/Opus封装且无ffmpeg；请安装ffmpeg或改为PCM WAV(8k/mono)'
                }

        # 设置API Key
        os.environ['DASHSCOPE_API_KEY'] = apikey or ''
        dashscope.api_key = apikey or ''

        # 统一采用实例+回调的方式
        class _EmptyCallback:
            def on_open(self):
                pass
            def on_event(self, result):
                pass
            def on_complete(self):
                pass
            def on_error(self, result):
                pass
            def on_close(self):
                pass

        rec = Recognition(
            callback=_EmptyCallback(),
            model="paraformer-realtime-8k-v2",
            sample_rate=sample_rate,
            format=call_format
        )
        result = rec.call(file=call_path)
    except Exception as e:
        return {
            'ok': False,
            'text': '',
            'code': 'EXCEPTION',
            'detail': str(e)
        }
    finally:
        # 还原API Key
        if prev_key_env is None:
            os.environ.pop('DASHSCOPE_API_KEY', None)
        else:
            os.environ['DASHSCOPE_API_KEY'] = prev_key_env
        dashscope.api_key = prev_key_attr or ''
        # 删除临时文件
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if tmp_wav_path and os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)
        except Exception:
            pass

    # 解析结果：优先从 get_sentence() 提取文本
    text_parts = []
    try:
        if hasattr(result, 'get_sentence'):
            sentences = result.get_sentence()
            if isinstance(sentences, list):
                for s in sentences:
                    if isinstance(s, dict) and 'text' in s:
                        text_parts.append(s['text'])
            elif isinstance(sentences, dict):
                if 'text' in sentences:
                    text_parts.append(sentences['text'])
    except Exception:
        pass

    final_text = ''.join(text_parts).strip()
    if final_text:
        return {
            'ok': True,
            'text': final_text,
            'code': 'ASR_OK',
            'detail': None
        }

    # 兜底：尽量返回可读信息
    try:
        resp = getattr(result, 'get_response', None)
        if callable(resp):
            return {
                'ok': False,
                'text': '',
                'code': 'SDK_NO_OUTPUT',
                'detail': {
                    'format': call_format,
                    'sample_rate': sample_rate,
                    'ffmpeg': bool(has_ffmpeg),
                    'response': str(resp())
                }
            }
    except Exception:
        pass
    try:
        return {
            'ok': False,
            'text': '',
            'code': 'SDK_NO_OUTPUT',
            'detail': {
                'format': call_format,
                'sample_rate': sample_rate,
                'ffmpeg': bool(has_ffmpeg),
                'result': str(result)
            }
        }
    except Exception as e:
        return {
            'ok': False,
            'text': '',
            'code': 'RESULT_SERIALIZE_ERROR',
            'detail': str(e)
        }
