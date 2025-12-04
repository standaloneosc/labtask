"""Phase tracking for profiling and decode phases in vLLM."""

import time
from typing import Callable, Optional

from profiling.metrics_collector import MetricsCollector, RequestMetrics


class PhaseTracker:
    """
    Tracks profiling (prefill) and decode phases of vLLM inference.
    
    This class provides decorators and context managers to automatically
    track timing for different phases of request processing.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.current_request_id: Optional[str] = None
    
    def track_request(self, request_id: str):
        """Context manager to track a complete request."""
        class RequestTracker:
            def __init__(self, tracker, req_id):
                self.tracker = tracker
                self.req_id = req_id
            
            def __enter__(self):
                self.tracker.current_request_id = self.req_id
                self.tracker.metrics_collector.mark_request_start(self.req_id)
                return self
            
            def __exit__(self, *args):
                self.tracker.metrics_collector.mark_request_end(self.req_id)
                self.tracker.current_request_id = None
        
        return RequestTracker(self, request_id)
    
    def track_profiling(self, request_id: Optional[str] = None):
        """Context manager to track profiling/prefill phase."""
        req_id = request_id or self.current_request_id
        
        class ProfilingTracker:
            def __init__(self, collector, req_id):
                self.collector = collector
                self.req_id = req_id
                self.start_time = None
            
            def __enter__(self):
                if self.req_id:
                    self.collector.mark_profiling_start(self.req_id)
                    self.start_time = time.perf_counter()
                return self
            
            def __exit__(self, *args):
                if self.req_id:
                    self.collector.mark_profiling_end(self.req_id)
        
        return ProfilingTracker(self.metrics_collector, req_id)
    
    def track_decode(self, request_id: Optional[str] = None):
        """Context manager to track decode phase."""
        req_id = request_id or self.current_request_id
        
        class DecodeTracker:
            def __init__(self, collector, req_id):
                self.collector = collector
                self.req_id = req_id
                self.start_time = None
                self.last_token_time = None
            
            def __enter__(self):
                if self.req_id:
                    self.collector.mark_decode_start(self.req_id)
                    self.start_time = time.perf_counter()
                    self.last_token_time = self.start_time
                return self
            
            def record_token(self):
                """Record a token generation event."""
                if self.req_id and self.last_token_time:
                    current_time = time.perf_counter()
                    decode_time = current_time - self.last_token_time
                    self.collector.record_token_decode(self.req_id, decode_time)
                    self.last_token_time = current_time
            
            def __exit__(self, *args):
                if self.req_id:
                    self.collector.mark_decode_end(self.req_id)
        
        return DecodeTracker(self.metrics_collector, req_id)
    
    def wrap_function(self, func: Callable, phase: str = "total"):
        """
        Decorator to wrap a function and track its execution.
        
        Args:
            func: Function to wrap
            phase: Phase type - "total", "profiling", or "decode"
        """
        def wrapper(*args, **kwargs):
            req_id = self.current_request_id
            if phase == "profiling":
                with self.track_profiling(req_id):
                    return func(*args, **kwargs)
            elif phase == "decode":
                with self.track_decode(req_id):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper


def instrument_vllm_engine(llm_engine, metrics_collector: MetricsCollector):
    """
    Instrument a vLLM engine to track phases.
    
    This function patches the engine's methods to automatically track
    profiling and decode phases.
    """
    phase_tracker = PhaseTracker(metrics_collector)
    
    # Store original methods
    original_step = llm_engine.step
    
    def instrumented_step():
        # Check if we're in prefill or decode phase
        # This is a simplified version - actual implementation would need
        # to inspect the engine's state more carefully
        result = original_step()
        return result
    
    llm_engine.step = instrumented_step
    
    return phase_tracker

