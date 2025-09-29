# Copilot Instructions for ChallengeCup_ChatDemo

## 项目概览
本项目为“数控机床AI语音助手”Web应用，基于 Flask + Flask-SocketIO 实现实时对话，集成讯飞星火大模型 API 进行智能问答，并支持本地 TTS 语音播报。另包含基于 LSTM 的机床故障预测模型训练脚本。

## 主要结构
- `assistant.py`：主后端服务，包含 Flask 路由、SocketIO 事件、星火 API 调用、TTS 语音播报、对话历史管理等。
- `templates/index.html`：前端页面，Socket.IO 实时通信，支持消息、日志、对话历史清空、AI思考中断等。
- `model_train.py`：数据处理与 LSTM 故障预测模型训练，输入为 `97.mat`（正常）和 `105.mat`（故障）数据。
- `cnc_fault_predictor.keras`：训练好的 Keras 模型文件。

## 关键开发/运行流程
- 启动服务：`python assistant.py`，访问 [http://localhost:5000](http://localhost:5000)
- 前端通过 Socket.IO 与后端实时通信，事件包括 `send_message`、`ai_response`、`clear_history`、`stop_generation` 等。
- 星火 API 调用需配置 `APPID`、`APIKey`、`APISecret`，详见 `assistant.py` 顶部。
- TTS 语音播报通过 pyttsx3，事件循环在独立线程中初始化，避免与 Flask 冲突。
- 对话历史保存在内存变量 `conversation_history`，每轮对话自动裁剪，防止 token 超限。
- 训练脚本 `model_train.py` 需本地有 `97.mat`、`105.mat`，输出模型为 `cnc_fault_predictor.keras`。

## 项目约定与模式
- 所有 AI 问答均以“你是数控机床运维专家”身份，回复风格简洁、专业、口语化。
- WebSocket 回调与 Flask 路由分离，AI回复/日志均通过 SocketIO 事件推送前端。
- 训练数据与模型文件与主服务分离，推理/预测功能未直接集成到 Web 服务。
- 仅支持中文界面与交互。

## 常见问题与调试
- 若 API 密钥错误或网络异常，AI 回复会推送错误日志到前端日志区。
- TTS 线程异常时会自动降级为主线程朗读。
- 端口/调试参数可在 `assistant.py` 末尾修改。

## 参考文件
- `assistant.py`：后端主逻辑与集成点
- `templates/index.html`：前端交互与事件绑定
- `model_train.py`：数据处理与模型训练流程

如需扩展模型推理、API 接入或多语言支持，请参考现有结构与事件流。
