"""Metrics collection for vLLM performance profiling."""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    
    request_id: str
    prompt_length: int
    output_length: int
    
    # Timing metrics
    arrival_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    total_latency: float = 0.0
    
    # Phase-specific metrics
    profiling_start: Optional[float] = None
    profiling_end: Optional[float] = None
    profiling_latency: Optional[float] = None
    
    decode_start: Optional[float] = None
    decode_end: Optional[float] = None
    decode_latency: Optional[float] = None
    
    # Per-token decode times
    decode_times: List[float] = field(default_factory=list)
    
    # Token generation metrics
    time_to_first_token: Optional[float] = None
    inter_token_latency: Optional[float] = None
    
    # Resource usage (sampled)
    cpu_usage_samples: List[float] = field(default_factory=list)
    gpu_usage_samples: List[float] = field(default_factory=list)
    gpu_memory_samples: List[float] = field(default_factory=list)
    
    def finalize(self) -> None:
        """Calculate derived metrics."""
        if self.start_time and self.end_time:
            self.total_latency = self.end_time - self.start_time
        
        if self.profiling_start and self.profiling_end:
            self.profiling_latency = self.profiling_end - self.profiling_start
        
        if self.decode_start and self.decode_end:
            self.decode_latency = self.decode_end - self.decode_start
        
        if self.decode_times:
            self.inter_token_latency = np.mean(self.decode_times)
            if len(self.decode_times) > 0:
                self.time_to_first_token = self.decode_times[0] if self.decode_start else None
        
        # Convert to dict for easier serialization
        self.decode_times = [float(t) for t in self.decode_times]
        self.cpu_usage_samples = [float(u) for u in self.cpu_usage_samples]
        self.gpu_usage_samples = [float(u) for u in self.gpu_usage_samples]
        self.gpu_memory_samples = [float(m) for m in self.gpu_memory_samples]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        self.finalize()
        return {
            "request_id": self.request_id,
            "prompt_length": self.prompt_length,
            "output_length": self.output_length,
            "arrival_time": self.arrival_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_latency": self.total_latency,
            "profiling_latency": self.profiling_latency,
            "decode_latency": self.decode_latency,
            "time_to_first_token": self.time_to_first_token,
            "inter_token_latency": self.inter_token_latency,
            "num_decode_tokens": len(self.decode_times),
            "avg_cpu_usage": float(np.mean(self.cpu_usage_samples)) if self.cpu_usage_samples else None,
            "avg_gpu_usage": float(np.mean(self.gpu_usage_samples)) if self.gpu_usage_samples else None,
            "max_gpu_memory_mb": float(np.max(self.gpu_memory_samples)) if self.gpu_memory_samples else None,
        }


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self):
        self.requests: Dict[str, RequestMetrics] = {}
        self.global_start_time: Optional[float] = None
        self.global_end_time: Optional[float] = None
        
    def start_collection(self) -> None:
        """Start global metrics collection."""
        self.global_start_time = time.perf_counter()
    
    def stop_collection(self) -> None:
        """Stop global metrics collection."""
        self.global_end_time = time.perf_counter()
    
    def create_request(self, request_id: str, prompt_length: int, 
                      output_length: int, arrival_time: Optional[float] = None) -> RequestMetrics:
        """Create a new request metrics object."""
        if arrival_time is None:
            arrival_time = time.perf_counter()
        
        metrics = RequestMetrics(
            request_id=request_id,
            prompt_length=prompt_length,
            output_length=output_length,
            arrival_time=arrival_time
        )
        self.requests[request_id] = metrics
        return metrics
    
    def get_request(self, request_id: str) -> Optional[RequestMetrics]:
        """Get metrics for a request."""
        return self.requests.get(request_id)
    
    def mark_request_start(self, request_id: str) -> None:
        """Mark when request processing starts."""
        metrics = self.get_request(request_id)
        if metrics:
            metrics.start_time = time.perf_counter()
    
    def mark_request_end(self, request_id: str) -> None:
        """Mark when request processing ends."""
        metrics = self.get_request(request_id)
        if metrics:
            metrics.end_time = time.perf_counter()
    
    def mark_profiling_start(self, request_id: str) -> None:
        """Mark when profiling phase starts."""
        metrics = self.get_request(request_id)
        if metrics:
            metrics.profiling_start = time.perf_counter()
    
    def mark_profiling_end(self, request_id: str) -> None:
        """Mark when profiling phase ends."""
        metrics = self.get_request(request_id)
        if metrics:
            metrics.profiling_end = time.perf_counter()
    
    def mark_decode_start(self, request_id: str) -> None:
        """Mark when decode phase starts."""
        metrics = self.get_request(request_id)
        if metrics:
            metrics.decode_start = time.perf_counter()
    
    def mark_decode_end(self, request_id: str) -> None:
        """Mark when decode phase ends."""
        metrics = self.get_request(request_id)
        if metrics:
            metrics.decode_end = time.perf_counter()
    
    def record_token_decode(self, request_id: str, decode_time: float) -> None:
        """Record time taken to decode a single token."""
        metrics = self.get_request(request_id)
        if metrics:
            metrics.decode_times.append(decode_time)
    
    def record_resource_sample(self, request_id: str, cpu: float, 
                               gpu: float, gpu_memory: float) -> None:
        """Record resource usage sample."""
        metrics = self.get_request(request_id)
        if metrics:
            metrics.cpu_usage_samples.append(cpu)
            metrics.gpu_usage_samples.append(gpu)
            metrics.gpu_memory_samples.append(gpu_memory)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics across all requests."""
        if not self.requests:
            return {}
        
        # Finalize all requests
        for metrics in self.requests.values():
            metrics.finalize()
        
        latencies = [m.total_latency for m in self.requests.values() if m.total_latency]
        profiling_latencies = [m.profiling_latency for m in self.requests.values() 
                              if m.profiling_latency is not None]
        decode_latencies = [m.decode_latency for m in self.requests.values() 
                           if m.decode_latency is not None]
        
        def compute_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            arr = np.array(values)
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
        
        total_time = (self.global_end_time - self.global_start_time 
                     if self.global_start_time and self.global_end_time else 0)
        
        total_tokens = sum(m.output_length for m in self.requests.values())
        total_prompt_tokens = sum(m.prompt_length for m in self.requests.values())
        
        return {
            "total_requests": len(self.requests),
            "total_time": total_time,
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "throughput_requests_per_sec": len(self.requests) / total_time if total_time > 0 else 0,
            "throughput_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
            "total_latency_stats": compute_stats(latencies),
            "profiling_latency_stats": compute_stats(profiling_latencies),
            "decode_latency_stats": compute_stats(decode_latencies),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics to dictionary."""
        summary = self.get_summary_statistics()
        requests_data = {rid: metrics.to_dict() 
                        for rid, metrics in self.requests.items()}
        
        return {
            "summary": summary,
            "requests": requests_data,
            "global_start_time": self.global_start_time,
            "global_end_time": self.global_end_time,
        }
    
    def save_json(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

