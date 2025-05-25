#!/usr/bin/env python3
"""
測試Streamlit按鈕狀態管理功能
"""

import streamlit as st
import time
import threading
from datetime import datetime

# 模擬訓練狀態
if 'training_status' not in st.session_state:
    st.session_state.training_status = 'idle'
if 'training_thread' not in st.session_state:
    st.session_state.training_thread = None
if 'stop_signal' not in st.session_state:
    st.session_state.stop_signal = False

def simulate_training():
    """模擬訓練過程"""
    st.session_state.training_status = 'running'
    st.session_state.stop_signal = False
    
    # 模擬訓練10秒
    for i in range(10):
        if st.session_state.stop_signal:
            print(f"訓練在第{i+1}秒被停止")
            break
        time.sleep(1)
        print(f"訓練進行中...第{i+1}秒")
    
    # 訓練結束
    if st.session_state.stop_signal:
        st.session_state.training_status = 'idle'
        print("訓練已停止")
    else:
        st.session_state.training_status = 'completed'
        print("訓練完成")

def start_training():
    """開始訓練"""
    thread = threading.Thread(target=simulate_training)
    thread.daemon = True
    thread.start()
    st.session_state.training_thread = thread
    return True

def stop_training():
    """停止訓練"""
    st.session_state.stop_signal = True
    if st.session_state.training_thread and st.session_state.training_thread.is_alive():
        st.session_state.training_thread.join(timeout=2.0)
    st.session_state.training_status = 'idle'
    return True

def reset_state():
    """重置狀態"""
    if st.session_state.training_status == 'running':
        stop_training()
    st.session_state.training_status = 'idle'
    st.session_state.training_thread = None
    st.session_state.stop_signal = False

# Streamlit UI
st.title("按鈕狀態管理測試")

# 顯示當前狀態
status_colors = {
    'idle': '🔵',
    'running': '🟡', 
    'completed': '🟢'
}

current_status = st.session_state.training_status
st.markdown(f"**當前狀態**: {status_colors.get(current_status, '⚪')} {current_status}")

# 按鈕區域
col1, col2, col3 = st.columns(3)

with col1:
    # 開始按鈕 - 只在idle、completed狀態下可用
    can_start = current_status in ['idle', 'completed']
    if st.button("🚀 開始訓練", disabled=not can_start, key="start_btn"):
        if start_training():
            st.success("訓練已開始")
            st.rerun()

with col2:
    # 停止按鈕 - 只在running狀態下可用
    can_stop = current_status == 'running'
    if st.button("⏹️ 停止訓練", disabled=not can_stop, key="stop_btn"):
        if stop_training():
            st.success("訓練已停止")
            st.rerun()

with col3:
    # 重置按鈕 - 始終可用
    if st.button("🔄 重置", key="reset_btn"):
        reset_state()
        st.success("狀態已重置")
        st.rerun()

# 顯示按鈕狀態
st.markdown("---")
st.subheader("按鈕狀態說明")
st.markdown(f"""
- **開始訓練按鈕**: {'✅ 可用' if can_start else '❌ 禁用'}
- **停止訓練按鈕**: {'✅ 可用' if can_stop else '❌ 禁用'}
- **重置按鈕**: ✅ 始終可用
""")

# 自動刷新
if current_status == 'running':
    time.sleep(1)
    st.rerun()