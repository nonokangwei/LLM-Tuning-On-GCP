#!/bin/bash

# 获取当前时间
now=$(date +%s)

# 计算 24 小时后的时间
after=$((now + 86400))

# 休眠 24 小时
until [ $(date +%s) -ge $after ]; do
    sleep 1
done

# 执行后续操作
echo "24 hours already!"