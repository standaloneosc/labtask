"""Generate visualizations from benchmark results."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


class BenchmarkPlotter:
    """Creates visualizations from benchmark results."""
    
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_latency_distribution(self, results: Dict, save_path: Optional[str] = None):
        """Plot latency distribution."""
        metrics = results.get("detailed_metrics", {}).get("requests", {})
        
        if not metrics:
            print("No request-level metrics found for latency distribution")
            return
        
        latencies = []
        profiling_latencies = []
        decode_latencies = []
        
        for req_data in metrics.values():
            if req_data.get("total_latency"):
                latencies.append(req_data["total_latency"])
            if req_data.get("profiling_latency"):
                profiling_latencies.append(req_data["profiling_latency"])
            if req_data.get("decode_latency"):
                decode_latencies.append(req_data["decode_latency"])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        if latencies:
            axes[0].hist(latencies, bins=50, alpha=0.7, edgecolor='black')
            axes[0].set_xlabel("Total Latency (s)")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title("Total Latency Distribution")
            axes[0].axvline(np.mean(latencies), color='r', linestyle='--', 
                          label=f"Mean: {np.mean(latencies):.3f}s")
            axes[0].legend()
        
        if profiling_latencies:
            axes[1].hist(profiling_latencies, bins=50, alpha=0.7, edgecolor='black', color='green')
            axes[1].set_xlabel("Profiling Latency (s)")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title("Profiling Phase Latency")
            axes[1].axvline(np.mean(profiling_latencies), color='r', linestyle='--',
                          label=f"Mean: {np.mean(profiling_latencies):.3f}s")
            axes[1].legend()
        
        if decode_latencies:
            axes[2].hist(decode_latencies, bins=50, alpha=0.7, edgecolor='black', color='orange')
            axes[2].set_xlabel("Decode Latency (s)")
            axes[2].set_ylabel("Frequency")
            axes[2].set_title("Decode Phase Latency")
            axes[2].axvline(np.mean(decode_latencies), color='r', linestyle='--',
                          label=f"Mean: {np.mean(decode_latencies):.3f}s")
            axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "latency_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_throughput_comparison(self, results_list: List[Dict], save_path: Optional[str] = None):
        """Compare throughput across configurations."""
        configs = []
        throughputs = []
        
        for result in results_list:
            config = result.get("config", {})
            metrics = result.get("metrics", {})
            
            config_label = (
                f"{config.get('model', 'unknown')}\n"
                f"TP={config.get('tensor_parallel_size', 1)}, "
                f"PP={config.get('pipeline_parallel_size', 1)}"
            )
            configs.append(config_label)
            throughputs.append(metrics.get("throughput_requests_per_sec", 0))
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(configs)), throughputs, alpha=0.7, edgecolor='black')
        plt.xlabel("Configuration")
        plt.ylabel("Throughput (requests/sec)")
        plt.title("Throughput Comparison Across Configurations")
        plt.xticks(range(len(configs)), configs, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, throughputs)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughputs)*0.01,
                    f"{val:.2f}", ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "throughput_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_latency_vs_throughput(self, results_list: List[Dict], save_path: Optional[str] = None):
        """Plot latency vs throughput tradeoff."""
        latencies = []
        throughputs = []
        labels = []
        
        for result in results_list:
            config = result.get("config", {})
            metrics = result.get("metrics", {})
            latency_stats = metrics.get("total_latency_stats", {})
            
            latency = latency_stats.get("mean", 0) if latency_stats else 0
            throughput = metrics.get("throughput_requests_per_sec", 0)
            
            latencies.append(latency)
            throughputs.append(throughput)
            
            label = (
                f"{config.get('model', 'unknown')}\n"
                f"TP={config.get('tensor_parallel_size', 1)}"
            )
            labels.append(label)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latencies, throughputs, s=200, alpha=0.6, edgecolors='black')
        
        # Annotate points
        for i, label in enumerate(labels):
            plt.annotate(label, (latencies[i], throughputs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel("Mean Latency (s)")
        plt.ylabel("Throughput (requests/sec)")
        plt.title("Latency vs Throughput Tradeoff")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "latency_vs_throughput.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_resource_usage(self, results: Dict, save_path: Optional[str] = None):
        """Plot resource usage over time."""
        resource_usage = results.get("resource_usage", {})
        
        if not resource_usage:
            print("No resource usage data found")
            return
        
        # Plot GPU utilization if available
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # GPU utilization
        gpu_data = {}
        for key, value in resource_usage.items():
            if key.startswith("gpu_") and isinstance(value, dict):
                util = value.get("utilization", {})
                if util:
                    gpu_data[key] = util.get("mean", 0)
        
        if gpu_data:
            axes[0].bar(gpu_data.keys(), gpu_data.values(), alpha=0.7, edgecolor='black')
            axes[0].set_ylabel("GPU Utilization (%)")
            axes[0].set_title("Average GPU Utilization")
            axes[0].set_ylim(0, 100)
            axes[0].grid(True, alpha=0.3, axis='y')
        
        # Memory usage
        memory = resource_usage.get("memory", {})
        if memory:
            mem_pct = memory.get("mean_percent", 0)
            axes[1].barh(["Memory"], [mem_pct], alpha=0.7, edgecolor='black')
            axes[1].set_xlabel("Memory Usage (%)")
            axes[1].set_title("Average Memory Usage")
            axes[1].set_xlim(0, 100)
            axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "resource_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_phase_comparison(self, results_list: List[Dict], save_path: Optional[str] = None):
        """Compare profiling vs decode phase times."""
        configs = []
        profiling_times = []
        decode_times = []
        
        for result in results_list:
            config = result.get("config", {})
            metrics = result.get("metrics", {})
            
            config_label = f"{config.get('model', 'unknown')}\nTP={config.get('tensor_parallel_size', 1)}"
            configs.append(config_label)
            
            profiling_stats = metrics.get("profiling_latency_stats", {})
            decode_stats = metrics.get("decode_latency_stats", {})
            
            profiling_times.append(profiling_stats.get("mean", 0) if profiling_stats else 0)
            decode_times.append(decode_stats.get("mean", 0) if decode_stats else 0)
        
        x = np.arange(len(configs))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, profiling_times, width, label='Profiling Phase', alpha=0.7)
        bars2 = ax.bar(x + width/2, decode_times, width, label='Decode Phase', alpha=0.7)
        
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Latency (s)")
        ax.set_title("Profiling vs Decode Phase Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "phase_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_plots(self, results: Dict, results_list: Optional[List[Dict]] = None):
        """Generate all standard plots."""
        print("Generating visualizations...")
        
        # Single result plots
        self.plot_latency_distribution(results)
        self.plot_resource_usage(results)
        
        # Comparison plots if multiple results
        if results_list and len(results_list) > 1:
            self.plot_throughput_comparison(results_list)
            self.plot_latency_vs_throughput(results_list)
            self.plot_phase_comparison(results_list)
        
        print(f"Visualizations saved to: {self.output_dir}")

