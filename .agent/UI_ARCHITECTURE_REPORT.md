# UI Architecture Report - Oanda Trading Bot

**Date**: 2025-11-25  
**Status**: Comprehensive Analysis Complete

---

## 🎯 UI Systems Overview

The project has **THREE DIFFERENT UI systems**, not one:

### 1. Training System UI (Streamlit) ✅
- **Location**: `src/oanda_trading_bot/training_system/app.py`
- **Technology**: Streamlit (Python-based)
- **Port**: 8501
- **Status**: EXISTS
- **Command**: `streamlit run src/oanda_trading_bot/training_system/app.py`

### 2. Live Trading Web UI (FastAPI + HTML/JS) ✅
- **Location**: `src/oanda_trading_bot/live_trading_system/web_ui/main.py`
- **Technology**: FastAPI + vanilla HTML/JS/CSS
- **Port**: 8000
- **Files**:
  - Backend: `web_ui/main.py` (FastAPI)
  - Frontend: `ui/index.html`, `ui/main.js`, `ui/styles.css`
- **Status**: EXISTS
- **Command**: `python src/oanda_trading_bot/live_trading_system/web_ui/main.py` or `uvicorn oanda_trading_bot.live_trading_system.web_ui.main:app`

### 3. Frontend React App (Under Development) ⚠️ 
- **Location**: `frontend/`
- **Technology**: React 18 + TypeScript + Material-UI (NOT Next.js, NOT Tailwind)
- **Port**: 3000 (default React)
- **Status**: PARTIAL - Code exists but may not be fully implemented
- **Package Manager**: npm/yarn
- **Command**: `cd frontend && npm start`

---

## 🔍 Technology Stack Details

### Frontend React App
```json
{
  "framework": "React 18.2.0 (create-react-app)",
  "language": "TypeScript",
  "ui_library": "Material-UI v5 (MUI)",
  "styling": "@emotion/styled (NOT Tailwind CSS)",
  "state": "Redux Toolkit",
  "charts": "lightweight-charts",
  "NOT_USING": ["Next.js", "Tailwind CSS"]
}
```

### Training System
```
Technology: Streamlit
Language: Python
Purpose: Model training interface
```

### Live Trading UI
```
Backend: FastAPI (Python)
Frontend: Vanilla HTML/JS/CSS
Purpose: Real-time trading dashboard
```

---

## ❌ Problems Found in quick_start.bat

### Issue 1: Live Trading UI Path is WRONG
```batch
# CURRENT (WRONG):
cd src\oanda_trading_bot\live_trading_system\ui
streamlit run app.py --server.port 8502

# PROBLEM:
- live_trading_system/ui/ contains HTML/JS files, NOT Streamlit
- There is NO app.py in that directory
```

### Issue 2: Missing Frontend Option
- The React frontend is not included in quick_start.bat at all

---

## ✅ Correct Commands

### 1. Training System UI
```batch
streamlit run src\oanda_trading_bot\training_system\app.py
# Port: 8501
# Access: http://localhost:8501
```

### 2. Live Trading Web UI
```batch
# Option A (Direct Python)
python src\oanda_trading_bot\live_trading_system\web_ui\main.py

# Option B (Uvicorn - Better)
uvicorn oanda_trading_bot.live_trading_system.web_ui.main:app --reload
# Port: 8000
# Access: http://localhost:8000
```

### 3. React Frontend (if needed)
```batch
cd frontend
npm install  # First time only
npm start
# Port: 3000
# Access: http://localhost:3000
```

---

## 🔧 Recommended quick_start.bat Structure

```batch
echo [1] Start API Server (Port 8000)
echo [2] Start Training System UI (Streamlit - Port 8501)
echo [3] Start Live Trading Web UI (FastAPI - Port 8000)
echo [4] Start React Frontend (Port 3000) [Development]
echo [5] Run Smoke Test
echo [6] Train Mini Model
echo [7] Check System Status
echo [0] Exit
```

---

## 📊 Port Allocation

| Service | Port | Technology |
|---------|------|------------|
| API Backend (NEW v2.0) | 8000 | FastAPI |
| Live Trading Web UI | 8000 | FastAPI (same as API) |
| Training UI | 8501 | Streamlit |
| React Frontend | 3000 | React (dev server) |

**CONFLICT**: Live Trading Web UI and API Backend both use port 8000!

---

## 🎯 Recommendations

### Immediate Fixes

1. **Fix quick_start.bat option [3]**
   ```batch
   :live
   echo Starting Live Trading Web UI...
   echo Access at: http://localhost:8000
   python src\oanda_trading_bot\live_trading_system\web_ui\main.py
   ```

2. **Add React Frontend option**
   ```batch
   :frontend
   echo Starting React Frontend (Development)...
   echo Access at: http://localhost:3000
   cd frontend
   npm start
   ```

3. **Document UI types clearly**
   - Make it clear there are 3 different UIs
   - Explain when to use each

### Long-term Improvements

1. **Unify UI ports**
   - Consider merging Live Web UI into the unified API backend
   - Use different ports for each service

2. **Complete React Frontend**
   - The React app seems incomplete
   - Should connect to unified API (port 8000)

3. **Remove redundant UIs**
   - Do you need BOTH Streamlit AND React?
   - Consider deprecating one

---

## 🚨 Critical Discovery

**The unified API backend (v2.0) we created earlier is SEPARATE from the Live Trading Web UI!**

- **Unified API**: `src/oanda_trading_bot/api/main.py` (Port 8000)
- **Live Web UI**: `src/oanda_trading_bot/live_trading_system/web_ui/main.py` (Port 8000)

**These two should be merged or use different ports!**

---

## ✅ Corrected File Paths

All paths verified:

```
✓ src/oanda_trading_bot/training_system/app.py
✓ src/oanda_trading_bot/live_trading_system/web_ui/main.py
✓ src/oanda_trading_bot/live_trading_system/ui/index.html
✓ src/oanda_trading_bot/api/main.py
✓ frontend/package.json
✓ frontend/src/
✗ src/oanda_trading_bot/live_trading_system/ui/app.py (DOES NOT EXIST)
```

---

## 📝 Summary

1. **You DON'T use Next.js** - You use Create React App
2. **You DON'T use Tailwind CSS** - You use Material-UI + Emotion
3. **You have 3 UIs**:
   - Streamlit (Training)
   - FastAPI + HTML (Live Trading)  
   - React + MUI (Frontend - partial)
4. **quick_start.bat needs fixing** - Wrong path for live trading UI
5. **Port conflict** - Two services on port 8000

---

*Analysis complete. Ready to fix quick_start.bat*
