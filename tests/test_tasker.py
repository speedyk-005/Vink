import time
import pytest

from vink.tasker import Tasker


def test_tasker_runs_once():
    """Test that Tasker runs task once when once=True."""
    result = {"done": False}
    
    def my_task():
        time.sleep(0.1)
        result["done"] = True
        return {"success": True}
    
    tasker = Tasker(task=my_task, once=True)
    
    assert tasker.lastest_result is None, "Result should be None before task runs"
    tasker.run()
    
    time.sleep(0.3)
    
    assert result["done"] is True, "Task should have been executed"
    assert tasker.lastest_result == {"success": True}, "Result should match task return value"

    with pytest.raises(ValueError, match="The worker has already stopped"):
        tasker.run()


def test_tasker_runs_multiple_times():
    """Test that Tasker can run multiple times when once=False."""
    counter = {"value": 0}
    
    def my_task():
        counter["value"] += 1
        return {"count": counter["value"]}
    
    tasker = Tasker(task=my_task, once=False)
    
    tasker.run()
    time.sleep(0.1)
    assert tasker.lastest_result == {"count": 1}, "First run should return count=1"
    
    tasker.run()
    time.sleep(0.1)
    assert tasker.lastest_result == {"count": 2}, "Second run should return count=2"
