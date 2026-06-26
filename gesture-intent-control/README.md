# Gesture Intent Control

一个“个人手势意图模型”的原型项目：用摄像头采集你的手势轨迹，训练一个轻量模型理解你的动作意图，再把意图映射成电脑操作。

它和普通“固定手势识别”的区别是：你可以录自己的动作习惯，让模型慢慢适配你的手势节奏、幅度和表达方式。

## 第一版架构

```mermaid
flowchart LR
    A["摄像头"] --> B["MediaPipe 手部 21 点"]
    B --> C["归一化 + 时序特征"]
    C --> D["个人意图模型"]
    D --> E["置信度 / 安全策略"]
    E --> F["电脑操作映射"]
```

## 能做什么

当前原型支持训练这些意图标签：

- `click`
- `double_click`
- `right_click`
- `scroll_up`
- `scroll_down`
- `nav_back`
- `nav_forward`
- `cancel`
- `pause`

你也可以在 `configs/intents.yaml` 里加自己的标签。

## 安装

建议使用 Python 3.10 或 3.11。

```bash
cd /Users/elevensum/Documents/Codex/2026-06-27/ni/outputs/gesture-intent-control
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS 上如果要真实控制电脑，需要给 Terminal / Codex / Python 授权：

- 摄像头权限
- 辅助功能权限：系统设置 → 隐私与安全性 → 辅助功能

## 1. 录制你的手势数据

每个意图先录 20～50 条，越多越稳。先从 3～4 个意图开始，不要一口气做太大。

```bash
python -m gesture_intent_control record --label click --seconds 1.2 --repeats 30
python -m gesture_intent_control record --label scroll_down --seconds 1.2 --repeats 30
python -m gesture_intent_control record --label nav_back --seconds 1.2 --repeats 30
python -m gesture_intent_control record --label cancel --seconds 1.2 --repeats 30
```

录制窗口里：

- 按 `space` 开始录一条样本
- 按 `q` 退出

数据会保存在 `data/raw/<label>/`。

## 2. 训练模型

```bash
python -m gesture_intent_control train
```

训练完成后会生成：

```text
models/intent_model.joblib
```

## 3. 实时运行

先用安全模式看识别结果，不会真的控制电脑：

```bash
python -m gesture_intent_control run
```

确认识别稳定后，再执行真实操作：

```bash
python -m gesture_intent_control run --execute
```

## 推荐的训练路线

第一阶段不要追求“全能手势系统”，先训练一个很顺的个人操作语言：

1. `click`：捏一下
2. `scroll_down`：手掌向上/向下推一下，看你自己习惯
3. `nav_back`：向左轻甩
4. `cancel`：张手停住或握拳

等这四个稳定后，再加：

- `double_click`
- `right_click`
- `nav_forward`
- `pause`

## 为什么不用端到端视频大模型？

第一版不建议直接拿视频帧训练 CNN/Transformer。原因很简单：你的样本不会很多，而手势控制最重要的是低延迟、稳定、安全。

这个项目先使用 MediaPipe 提取手部关键点，再训练轻量分类器。好处是：

- 数据需求小
- 训练很快
- 本地实时运行
- 更容易调试“模型为什么误判”

等你有了几千条个人样本之后，再升级到 LSTM/TCN/Transformer 会更自然。

## 文件结构

```text
gesture-intent-control/
  configs/intents.yaml          # 意图到电脑动作的映射
  data/raw/                     # 录制的手势样本
  models/                       # 训练后的模型
  src/gesture_intent_control/
    mediapipe_hand.py           # 摄像头手部关键点
    features.py                 # 手势时序特征
    recorder.py                 # 数据采集
    train.py                    # 训练模型
    realtime.py                 # 实时推理
    actions.py                  # 电脑控制映射
```

## 下一步可升级方向

- 加一个“纠错键”：模型识别错了，你按键纠正，自动加入训练集。
- 加上下文：当前是浏览器、编辑器还是 Finder，不同 app 下同一手势触发不同动作。
- 加两手手势：缩放、窗口切换、空间拖拽。
- 把连续控制拆出来：鼠标移动和拖拽更适合规则/回归模型，离散意图适合分类模型。

