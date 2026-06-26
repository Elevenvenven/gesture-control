# Gesture Control

用电脑内置摄像头识别手势，控制滚动、鼠标移动、点击和页面前进/后退。

## 安装

```bash
cd gesturepro
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

首次运行会自动下载 MediaPipe 手部识别模型（约 3 MB）到 `models/` 目录。

## 运行

```bash
python gesture_scroll.py
```

查看可用摄像头：

```bash
python gesture_scroll.py --list-cameras
```

在这台 Mac 上，OpenCV 的 `--camera 0` 是手机连续互通摄像头，`--camera 1` 对应电脑内置 FaceTime HD Camera。

## 手势

打开任意网页或文档，把一只手对着摄像头：

| 手势 | 动作 |
|------|------|
| 张开手掌上下滑 | 页面滚动 |
| 张开手掌左右滑 | 浏览器后退 / 前进 |
| 只伸食指 | 移动鼠标 |
| 拇指食指快速捏合 | 单击 |
| 拇指食指捏住不放 | 拖拽，松开后放下 |
| 食指 + 中指 | 右键 |
| 食指 + 中指 + 无名指 | 双击 |
| 握拳 | 暂停 / 恢复手势控制 |
| Q / Esc | 退出 |

## macOS 权限

第一次运行时，系统可能需要授权：

- **摄像头**：允许 Terminal、iTerm、VS Code 或 Cursor 使用摄像头
- **辅助功能**：允许 Python 或终端控制电脑（滚动、鼠标）

路径：`系统设置 -> 隐私与安全性 -> 摄像头 / 辅助功能`

## 调参

如果还是卡：

```bash
python gesture_scroll.py --model-quality fast
```

光标精度不够：

```bash
python gesture_scroll.py --model-quality accurate --landmark-alpha 0.55 --smooth-alpha 0.45
```

光标太飘：

```bash
python gesture_scroll.py --smooth-alpha 0.35 --landmark-alpha 0.5
```

滚动太敏感：

```bash
python gesture_scroll.py --threshold 0.12 --cooldown 0.35
```

反应太慢：

```bash
python gesture_scroll.py --threshold 0.06 --frames 5
```

## 项目结构

```
gesturepro/
├── gesture_scroll.py   # 主程序与 UI
├── gestures.py         # MediaPipe Tasks 手部追踪与手势分类
├── controller.py       # 鼠标/键盘控制与光标平滑
├── models/             # 自动下载的模型文件
└── requirements.txt
```

## 升级说明（v3 — 精准识别）

- **One Euro 滤波**：关键点平滑，慢动稳、快动跟手
- **手指置信度评分**：综合关节角度、距离、深度、手掌朝向，0–1 分
- **时序投票**：连续 7 帧加权投票，过滤单帧误判
- **捏合增强**：要求拇指伸展 + 连续 2 帧确认
- **光标混合定位**：食指尖 78% + 指关节 22%，指向更稳
- 画面左下角显示 **手势名称 + 置信度%**
- 精准模式推理分辨率提升至 **720px**

若误触仍多，提高置信度门槛：

```bash
python gesture_scroll.py --min-confidence 0.65
```
