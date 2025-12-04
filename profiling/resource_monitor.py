"""Resource monitoring for CPU and GPU usage during benchmarking."""

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import psutil

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time."""
    timestamp: float
    cpu_percent: float
    cpu_count: int
    memory_total_mb: float
    memory_used_mb: float
    memory_percent: float
    gpu_utilizations: List[float]
    gpu_memory_used_mb: List[float]
    gpu_memory_total_mb: List[float]
    gpu_temperatures: List[float]


class ResourceMonitor:
    """Monitors CPU and GPU resource usage over time."""
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Initialize resource monitor.
        
        Args:
            sampling_interval: How often to sample resources (seconds)
        """
        self.sampling_interval = sampling_interval
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Initialize NVML if available
        self.nvml_initialized = False
        self.gpu_count = 0
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.nvml_initialized = True
            except Exception:
                pass
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.snapshots = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            snapshot = self._take_snapshot()
            self.snapshots.append(snapshot)
            time.sleep(self.sampling_interval)
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage."""
        timestamp = time.perf_counter()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_utilizations = []
        gpu_memory_used = []
        gpu_memory_total = []
        gpu_temperatures = []
        
        if self.nvml_initialized:
            for i in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilizations.append(float(util.gpu))
                    
                    # Memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used.append(float(mem_info.used) / 1024 / 1024)  # MB
                    gpu_memory_total.append(float(mem_info.total) / 1024 / 1024)  # MB
                    
                    # Temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        gpu_temperatures.append(float(temp))
                    except:
                        gpu_temperatures.append(0.0)
                except Exception:
                    gpu_utilizations.append(0.0)
                    gpu_memory_used.append(0.0)
                    gpu_memory_total.append(0.0)
                    gpu_temperatures.append(0.0)
        
        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            memory_total_mb=float(memory.total) / 1024 / 1024,
            memory_used_mb=float(memory.used) / 1024 / 1024,
            memory_percent=memory.percent,
            gpu_utilizations=gpu_utilizations,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_temperatures=gpu_temperatures,
        )
    
    def get_summary(self) -> Dict:
        """Get summary statistics of resource usage."""
        if not self.snapshots:
            return {}
        
        cpu_percents = [s.cpu_percent for s in self.snapshots]
        memory_percents = [s.memory_percent for s in self.snapshots]
        
        summary = {
            "duration": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            "num_samples": len(self.snapshots),
            "cpu": {
                "mean": float(np.mean(cpu_percents)),
                "max": float(np.max(cpu_percents)),
                "std": float(np.std(cpu_percents)),
            },
            "memory": {
                "mean_percent": float(np.mean(memory_percents)),
                "max_percent": float(np.max(memory_percents)),
                "max_used_mb": float(max(s.memory_used_mb for s in self.snapshots)),
            },
        }
        
        if self.gpu_count > 0 and self.snapshots:
            for gpu_idx in range(self.gpu_count):
                gpu_utils = [s.gpu_utilizations[gpu_idx] 
                           for s in self.snapshots 
                           if len(s.gpu_utilizations) > gpu_idx]
                gpu_mem = [s.gpu_memory_used_mb[gpu_idx]
                          for s in self.snapshots
                          if len(s.gpu_memory_used_mb) > gpu_idx]
                
                if gpu_utils:
                    summary[f"gpu_{gpu_idx}"] = {
                        "utilization": {
                            "mean": float(np.mean(gpu_utils)),
                            "max": float(np.max(gpu_utils)),
                            "std": float(np.std(gpu_utils)),
                        },
                        "memory": {
                            "mean_mb": float(np.mean(gpu_mem)),
                            "max_mb": float(np.max(gpu_mem)),
                            "std_mb": float(np.std(gpu_mem)),
                        },
                    }
        
        return summary
    
    def to_dict(self) -> Dict:
        """Export all snapshots to dictionary."""
        return {
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "cpu_percent": s.cpu_percent,
                    "memory_percent": s.memory_percent,
                    "gpu_utilizations": s.gpu_utilizations,
                    "gpu_memory_used_mb": s.gpu_memory_used_mb,
                }
                for s in self.snapshots
            ],
            "summary": self.get_summary(),
        }

