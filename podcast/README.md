# 播客生成工具 Podcast Generator

将文章/文本转换为双人播客对话，内置预设音色，无需录音即可使用。

## 工作流程

1. **选择声音模式** — 预设音色（无需录制，开箱即用）或自定义音色（上传/录制参考音频）
2. **对话生成** — DeepSeek API 将输入文本拆解为自然对话脚本
3. **语音合成** — MOSS-TTS-Nano 分别为两位说话人合成语音并拼接为完整播客

## 环境配置

```bash
conda create -n podcast python=3.12 -y
conda activate podcast
pip install openai gradio mlx-audio numpy
```

## 设置 API Key

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

或在 Web 界面的「API 配置」面板中输入。

## 使用

```bash
python app.py
```

打开 http://localhost:7865

### 步骤

1. 选择「预设音色」模式（默认），直接使用内置音色，无需录音或上传
2. 或切换到「自定义音色」模式，上传/录制两段参考音频（主持人 + 嘉宾，各 3-10 秒清晰人声）
3. 粘贴要转换为播客的文章/文本
4. 点击「生成对话脚本」，预览并可编辑对话
5. 点击「合成播客音频」，等待生成完成

## 预设音色

`voices/` 目录下内置了预设音色，预设模式下可直接使用：

- **磁性男声** — 适合主持人 (Speaker A)
- **清脆女声** — 适合嘉宾 (Speaker B)

如需自定义音色，请切换到自定义模式并上传/录制参考音频。

## 参考音频要求（自定义模式）

- 每位说话人 3-10 秒的清晰语音
- 背景安静、无杂音
- 建议使用正常语速和语调
- 支持中英等 20 种语言

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Temperature | 0.8 | 语音合成的随机性 |
| Top-p | 0.95 | 核采样阈值 |
| Top-k | 25 | Top-k 采样 |
| Max Tokens | 375 | 每句最大生成 token 数 |
| Repetition Penalty | 1.2 | 重复惩罚系数 |
| 句间静音 | 0.3s | 对话之间的停顿长度 |

## 技术栈

- [MOSS-TTS-Nano (100M)](https://huggingface.co/mlx-community/MOSS-TTS-Nano-100M) — MLX TTS 声音克隆
- [DeepSeek API](https://api.deepseek.com) — 对话脚本生成
- [Gradio](https://gradio.app) — Web UI
