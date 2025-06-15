#!/usr/bin/env python3
"""
Simple Test Runner - Progressive testing from simple to complex
"""
import os
import sys
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_test(test_file):
    """Run a single test file"""
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"ERROR: Test file not found: {test_file}")
        return False
    
    print(f"\n[RUNNING] {test_file}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=project_root
        )
        
        if result.returncode == 0:
            print(f"[PASSED] {test_file}")
            if result.stdout:
                print("Output:", result.stdout.strip())
            return True
        else:
            print(f"[FAILED] {test_file}")
            if result.stderr:
                print("Error:", result.stderr.strip())
            if result.stdout:
                print("Output:", result.stdout.strip())
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {test_file}")
        return False
    except Exception as e:
        print(f"[ERROR] {test_file}: {e}")
        return False

def main():
    """Main test runner"""
    print("OANDA Trading Bot - Progressive Test Runner")
    print("=" * 60)
    
    # Test stages
    stages = {
        "Stage 1 - Basic Components": [
            "tests/unit_tests/test_config.py",
            "tests/unit_tests/test_data_structures.py", 
            "tests/unit_tests/test_basic_components.py"
        ],        "Stage 2 - Model Architecture": [
            "tests/unit_tests/test_enhanced_transformer.py",
            "tests/unit_tests/test_quantum_strategies_simple.py",
            "tests/unit_tests/test_model_components.py"
        ],        "Stage 3 - Agent and Environment": [
            "tests/unit_tests/test_sac_agent.py",
            "tests/unit_tests/test_trading_environment.py",
            "tests/unit_tests/test_reward_system_simple.py"
        ],        "Stage 4 - Integration Tests": [
            "tests/integration_tests/test_model_integration.py",
            "tests/integration_tests/test_agent_environment_integration.py",
            "tests/integration_tests/test_enhanced_gradient_flow_simple.py"
        ],        "Stage 5 - End-to-End Tests": [
            "tests/integration_tests/test_complete_training_flow_simple.py",
            "tests/performance_tests/test_training_performance_simple.py"
        ]
    }
    
    # Check which stage to run
    if len(sys.argv) > 1:
        stage_arg = sys.argv[1].lower()
        if stage_arg == "stage1":
            selected_stages = {"Stage 1 - Basic Components": stages["Stage 1 - Basic Components"]}
        elif stage_arg == "stage2":
            selected_stages = {"Stage 2 - Model Architecture": stages["Stage 2 - Model Architecture"]}
        elif stage_arg == "stage3":
            selected_stages = {"Stage 3 - Agent and Environment": stages["Stage 3 - Agent and Environment"]}
        elif stage_arg == "stage4":
            selected_stages = {"Stage 4 - Integration Tests": stages["Stage 4 - Integration Tests"]}
        elif stage_arg == "stage5":
            selected_stages = {"Stage 5 - End-to-End Tests": stages["Stage 5 - End-to-End Tests"]}
        else:
            selected_stages = stages
    else:
        selected_stages = stages
    
    # Run tests
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for stage_name, test_files in selected_stages.items():
        print(f"\n\n>>> {stage_name}")
        print("=" * 60)
        
        stage_passed = 0
        stage_total = len(test_files)
        
        for test_file in test_files:
            total_tests += 1
            if run_test(test_file):
                passed_tests += 1
                stage_passed += 1
            else:
                failed_tests += 1
        
        print(f"\n{stage_name} Results: {stage_passed}/{stage_total} passed")
        
        # Stop on failure if requested
        if failed_tests > 0 and "--stop-on-failure" in sys.argv:
            print(f"\nStopping due to failures in {stage_name}")
            break
        
        # Wait between stages
        if stage_name != list(selected_stages.keys())[-1]:
            print("\nWaiting 2 seconds before next stage...")
            time.sleep(2)
    
    # Final results
    print("\n\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    if failed_tests == 0:
        print("\nALL TESTS PASSED! System is working correctly.")
    else:
        print(f"\n{failed_tests} tests failed. Please check the errors above.")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")

if __name__ == "__main__":
    main()
