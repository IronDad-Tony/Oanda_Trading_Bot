# 自动化测试框架架构

## 目录结构
```mermaid
graph TD
    A[test_framework] --> B[sandbox]
    A --> C[model_validation]
    A --> D[trading_cycle]
    A --> E[utils]
    
    B --> B1[api_simulator.py]
    B --> B2[db_mocker.py]
    B --> B3[account_manager.py]
    
    C --> C1[model_loader_test.py]
    C --> C2[dimensional_validator.py]
    C --> C3[inference_benchmark.py]
    
    D --> D1[full_pipeline_test.py]
    D --> D2[edge_case_test.py]
    D --> D3[signal_consistency_check.py]
    
    E --> E1[cleanup_manager.py]
    E --> E2[report_generator.py]
```

## 核心模块
1. **沙盒环境初始化** (`sandbox/`)
   - `api_simulator.py`: 模拟Oanda API响应
   - `db_mocker.py`: 内存数据库实例
   - `account_manager.py`: 虚拟账户管理

2. **模型载入验证** (`model_validation/`)
   - 文件存在性检查
   - 输入/输出维度验证
   - 推理速度基准测试

3. **交易循环测试** (`trading_cycle/`)
   - 端到端流水线测试
   - 边界条件测试（空数据、极端值）
   - 信号一致性校验

4. **自动清理机制** (`utils/cleanup_manager.py`)
   - 资源释放
   - 临时文件清除
   - 数据库状态回滚