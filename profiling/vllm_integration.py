"""Integration helpers for extracting metrics from vLLM outputs."""

from typing import Any, Optional

from profiling.metrics_collector import MetricsCollector, RequestMetrics


def extract_vllm_metrics(
    output: Any,
    request_id: str,
    metrics_collector: MetricsCollector,
) -> None:
    """
    Extract timing metrics from vLLM RequestOutput.
    
    Args:
        output: vLLM RequestOutput object
        request_id: Request identifier
        metrics_collector: Metrics collector instance
    """
    request_metrics = metrics_collector.get_request(request_id)
    if not request_metrics:
        return
    
    # Check for vLLM metrics in output
    if not hasattr(output, 'metrics') or not output.metrics:
        return
    
    vllm_metrics = output.metrics
    
    # Try to extract from RequestStateStats (vLLM v1)
    # RequestStateStats has: prefill_time, decode_time, e2e_latency, etc.
    if hasattr(vllm_metrics, 'prefill_time') or hasattr(vllm_metrics, 'queued_time'):
        # This is likely RequestStateStats from vLLM v1
        try:
            # Prefill/profiling phase
            if hasattr(vllm_metrics, 'prefill_time') and vllm_metrics.prefill_time:
                prefill_time = float(vllm_metrics.prefill_time)
                request_metrics.profiling_latency = prefill_time
            
            # Decode phase
            if hasattr(vllm_metrics, 'decode_time') and vllm_metrics.decode_time:
                decode_time = float(vllm_metrics.decode_time)
                request_metrics.decode_latency = decode_time
                
                # Extract mean time per token if available
                if hasattr(vllm_metrics, 'mean_time_per_output_token'):
                    mean_time = float(vllm_metrics.mean_time_per_output_token)
                    if mean_time > 0:
                        # Estimate per-token times
                        num_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
                        for _ in range(num_tokens):
                            metrics_collector.record_token_decode(request_id, mean_time)
            
            # Time to first token
            if hasattr(vllm_metrics, 'queued_time') and hasattr(vllm_metrics, 'prefill_time'):
                queued_time = float(vllm_metrics.queued_time) if vllm_metrics.queued_time else 0
                prefill_time = float(vllm_metrics.prefill_time) if vllm_metrics.prefill_time else 0
                request_metrics.time_to_first_token = queued_time + prefill_time
            
            # End-to-end latency
            if hasattr(vllm_metrics, 'e2e_latency') and vllm_metrics.e2e_latency:
                request_metrics.total_latency = float(vllm_metrics.e2e_latency)
        
        except (ValueError, TypeError, AttributeError) as e:
            # Fallback: try to extract from RequestMetrics (vLLM v0)
            pass
    
    # Try to extract from RequestMetrics (vLLM v0/legacy)
    # RequestMetrics has: arrival_time, first_token_time, finished_time, etc.
    if hasattr(vllm_metrics, 'arrival_time'):
        try:
            arrival = vllm_metrics.arrival_time
            if arrival:
                request_metrics.arrival_time = float(arrival)
        except (ValueError, TypeError):
            pass
    
    if hasattr(vllm_metrics, 'first_token_time'):
        try:
            first_token = vllm_metrics.first_token_time
            start = request_metrics.start_time
            if first_token and start:
                request_metrics.time_to_first_token = float(first_token) - float(start)
        except (ValueError, TypeError):
            pass
    
    if hasattr(vllm_metrics, 'finished_time'):
        try:
            finished = vllm_metrics.finished_time
            start = request_metrics.start_time
            if finished and start:
                request_metrics.end_time = float(finished)
                request_metrics.total_latency = float(finished) - float(start)
        except (ValueError, TypeError):
            pass


def extract_token_timing_from_outputs(
    outputs: list,
    request_id: str,
    metrics_collector: MetricsCollector,
) -> None:
    """
    Extract per-token timing from streaming outputs.
    
    This is used when processing streaming outputs where we can
    track when each token arrives.
    """
    request_metrics = metrics_collector.get_request(request_id)
    if not request_metrics:
        return
    
    # If we have multiple outputs (streaming), calculate inter-token latency
    if len(outputs) > 1:
        decode_times = []
        last_time = request_metrics.decode_start
        
        for i, output in enumerate(outputs):
            if hasattr(output, 'metrics') and output.metrics:
                if hasattr(output.metrics, 'first_token_time'):
                    current_time = float(output.metrics.first_token_time)
                    if last_time:
                        decode_times.append(current_time - last_time)
                    last_time = current_time
        
        # Record decode times
        for decode_time in decode_times:
            metrics_collector.record_token_decode(request_id, decode_time)

