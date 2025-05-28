# setup_python_path.py
"""
Python 路徑設置工具
在導入項目模組之前運行此腳本以確保正確的模組路徑設置
"""
import sys
import os
from pathlib import Path

def setup_project_path():
    """設置項目的 Python 路徑"""
    # 獲取項目根目錄
    project_root = Path(__file__).resolve().parent
    
    # 將項目根目錄添加到 Python 路徑的開頭
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"已將項目根目錄添加到 Python 路徑: {project_root}")
    
    # 設置 PYTHONPATH 環境變量
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(project_root) not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{project_root}{os.pathsep}{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = str(project_root)
        print(f"已設置 PYTHONPATH 環境變量包含項目根目錄")
    
    return project_root

if __name__ == "__main__":
    setup_project_path()
    print("Python 路徑設置完成！")
    print(f"當前 sys.path 的前幾項:")
    for i, path in enumerate(sys.path[:5]):
        print(f"  {i}: {path}")
