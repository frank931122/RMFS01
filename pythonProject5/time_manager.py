# time_manager.py

class TimeManager:
    def __init__(self):
        self.global_time = 0  # 初始化全局时间为0

    def get_current_time(self):
        """返回当前全局时间"""
        return self.global_time

    def advance_time(self, time_units):
        """推进全局时间"""
        self.global_time += time_units
        print(f"[TimeManager] Global time advanced by {time_units} units. Current time: {self.global_time}")

    def reset_time(self):
        """重置全局时间"""
        self.global_time = 0
        print("[TimeManager] Global time reset to 0.")
