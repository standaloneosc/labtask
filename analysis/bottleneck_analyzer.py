"""Analyzes performance bottlenecks from benchmark results."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BottleneckAnalysis:
    """Results of bottleneck analysis."""
    primary_bottleneck: str
    bottleneck_severity: float  # 0-1 scale
    bottleneck_details: Dict
    recommendations: List[str]


class BottleneckAnalyzer:
    """Analyzes benchmark results to identify performance bottlenecks."""
    
    def analyze(self, benchmark_results: Dict) -> BottleneckAnalysis:
        """
        Analyze benchmark results to identify bottlenecks.
        
        Args:
            benchmark_results: Results from benchmark run
            
        Returns:
            BottleneckAnalysis with findings and recommendations
        """
        metrics = benchmark_results.get("metrics", {})
        resource_usage = benchmark_results.get("resource_usage", {})
        
        # Analyze different potential bottlenecks
        bottlenecks = []
        
        # 1. Check profiling phase vs decode phase
        profiling_stats = metrics.get("profiling_latency_stats", {})
        decode_stats = metrics.get("decode_latency_stats", {})
        
        if profiling_stats and decode_stats:
            profiling_pct = profiling_stats.get("mean", 0) / (
                profiling_stats.get("mean", 0) + decode_stats.get("mean", 0) + 1e-10
            )
            
            if profiling_pct > 0.5:
                bottlenecks.append({
                    "type": "profiling_phase",
                    "severity": profiling_pct,
                    "details": {
                        "profiling_time_pct": profiling_pct * 100,
                        "profiling_mean": profiling_stats.get("mean"),
                        "decode_mean": decode_stats.get("mean"),
                    },
                    "recommendation": (
                        "Profiling phase is taking majority of time. "
                        "Consider: optimizing prefill, using chunked prefill, "
                        "or reducing prompt lengths."
                    ),
                })
        
        # 2. Check GPU utilization
        if resource_usage:
            for key, value in resource_usage.items():
                if key.startswith("gpu_") and isinstance(value, dict):
                    util = value.get("utilization", {})
                    mean_util = util.get("mean", 0)
                    
                    if mean_util < 50:
                        bottlenecks.append({
                            "type": f"low_gpu_utilization_{key}",
                            "severity": 1 - (mean_util / 100),
                            "details": {
                                "gpu": key,
                                "utilization": mean_util,
                            },
                            "recommendation": (
                                f"GPU {key} has low utilization ({mean_util:.1f}%). "
                                "Consider: increasing batch size, using tensor parallelism, "
                                "or checking for CPU-bound operations."
                            ),
                        })
        
        # 3. Check memory pressure
        if resource_usage:
            memory = resource_usage.get("memory", {})
            max_memory_pct = memory.get("max_percent", 0)
            
            if max_memory_pct > 90:
                bottlenecks.append({
                    "type": "memory_pressure",
                    "severity": (max_memory_pct - 90) / 10,  # Normalize 90-100 to 0-1
                    "details": {
                        "max_memory_percent": max_memory_pct,
                    },
                    "recommendation": (
                        f"High memory usage ({max_memory_pct:.1f}%). "
                        "Consider: reducing batch size, using KV cache offloading, "
                        "or using quantization."
                    ),
                })
        
        # 4. Check latency distribution (high variance = instability)
        latency_stats = metrics.get("total_latency_stats", {})
        if latency_stats:
            mean_latency = latency_stats.get("mean", 0)
            std_latency = latency_stats.get("std", 0)
            
            if mean_latency > 0:
                cv = std_latency / mean_latency  # Coefficient of variation
                
                if cv > 0.5:
                    bottlenecks.append({
                        "type": "high_latency_variance",
                        "severity": min(cv, 1.0),
                        "details": {
                            "coefficient_of_variation": cv,
                            "mean_latency": mean_latency,
                            "std_latency": std_latency,
                        },
                        "recommendation": (
                            "High latency variance detected. "
                            "Consider: more stable arrival patterns, "
                            "larger batch sizes, or better scheduling."
                        ),
                    })
        
        # 5. Check throughput vs latency tradeoff
        throughput_rps = metrics.get("throughput_requests_per_sec", 0)
        mean_latency = latency_stats.get("mean", 0) if latency_stats else 0
        
        if throughput_rps > 0 and mean_latency > 0:
            # Low throughput relative to latency suggests batching issues
            expected_throughput = 1.0 / mean_latency if mean_latency > 0 else 0
            throughput_ratio = throughput_rps / expected_throughput if expected_throughput > 0 else 0
            
            if throughput_ratio < 0.5:
                bottlenecks.append({
                    "type": "batching_inefficiency",
                    "severity": 1 - throughput_ratio,
                    "details": {
                        "actual_throughput": throughput_rps,
                        "expected_throughput": expected_throughput,
                        "ratio": throughput_ratio,
                    },
                    "recommendation": (
                        "Throughput is lower than expected for given latency. "
                        "Consider: increasing batch size, using continuous batching, "
                        "or optimizing scheduler."
                    ),
                })
        
        # Select primary bottleneck
        if bottlenecks:
            primary = max(bottlenecks, key=lambda x: x["severity"])
        else:
            primary = {
                "type": "none",
                "severity": 0.0,
                "details": {},
                "recommendation": "No major bottlenecks detected.",
            }
        
        return BottleneckAnalysis(
            primary_bottleneck=primary["type"],
            bottleneck_severity=primary["severity"],
            bottleneck_details=primary["details"],
            recommendations=[b["recommendation"] for b in bottlenecks],
        )
    
    def compare_configurations(self, results_list: List[Dict]) -> Dict:
        """
        Compare multiple benchmark configurations to identify best settings.
        
        Args:
            results_list: List of benchmark result dictionaries
            
        Returns:
            Comparison analysis
        """
        comparison = {
            "configs": [],
            "best_config": None,
            "improvements": {},
        }
        
        best_throughput = 0
        best_latency = float('inf')
        best_config_idx = None
        
        for i, result in enumerate(results_list):
            config = result.get("config", {})
            metrics = result.get("metrics", {})
            
            throughput = metrics.get("throughput_requests_per_sec", 0)
            latency_stats = metrics.get("total_latency_stats", {})
            mean_latency = latency_stats.get("mean", float('inf')) if latency_stats else float('inf')
            
            comparison["configs"].append({
                "index": i,
                "model": config.get("model"),
                "tensor_parallel": config.get("tensor_parallel_size"),
                "pipeline_parallel": config.get("pipeline_parallel_size"),
                "throughput": throughput,
                "latency": mean_latency,
            })
            
            # Find best config (could use different criteria)
            if throughput > best_throughput:
                best_throughput = throughput
                best_config_idx = i
            
            if mean_latency < best_latency:
                best_latency = mean_latency
        
        comparison["best_config"] = best_config_idx
        comparison["throughput_improvement"] = (
            (best_throughput / comparison["configs"][0]["throughput"] - 1) * 100
            if comparison["configs"] else 0
        )
        
        return comparison

