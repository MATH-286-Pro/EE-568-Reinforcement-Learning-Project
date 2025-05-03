#!/usr/bin/env bash

# netstat -aon | findstr :6006


# LOGDIR="Project/Training/Logs/PPO_6"

# # taskkill //PID 20784 //F

# tensorboard --logdir "${LOGDIR}" --port 6006 --bind_all




# 要使用的端口和日志目录
PORT=6006
LOGDIR="Project/Training/Logs/PPO_13"

# 1) 找到占用端口的 LISTENING 行
LINE=$(netstat -aon | findstr ":${PORT}" | findstr LISTENING)

if [ -n "$LINE" ]; then
  # 2) 从这行输出里取出最后一列（PID）
  PID=$(echo $LINE | awk '{print $5}')
  echo "▶️ 端口 ${PORT} 被进程 ${PID} 占用，正在终止……"
  # 3) 杀掉它（Git-bash 下用 // 规避路径转换）
  taskkill //PID $PID //F
else
  echo "ℹ️ 端口 ${PORT} 当前未被占用"
fi

# 4) 启动 TensorBoard
echo "▶️ 启动 TensorBoard，日志目录：${LOGDIR}，端口：${PORT}"
tensorboard --logdir "${LOGDIR}" --port ${PORT} --bind_all
