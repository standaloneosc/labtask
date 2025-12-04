#!/usr/bin/env python3
"""
Main script to run comprehensive vLLM performance benchmarks.

This script orchestrates:
- Dataset preparation
- Load generation with configurable arrival patterns
- Benchmark execution across different configurations
- Performance analysis and bottleneck identification
- Visualization generation
"""

import argparse
import json
from pathlib import Path

from analysis.bottleneck_analyzer import BottleneckAnalyzer
from benchmarks.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from benchmarks.dataset_manager import DatasetManager
from benchmarks.load_generator import ArrivalPattern
from visualization.plotter import BenchmarkPlotter


def create_default_configs():
    """Create default benchmark configurations for testing."""
    configs = []
    
    # Base configuration
    base_model = "meta-llama/Llama-2-7b-hf"  # Example model
    
    # Configuration 1: Single GPU, default settings
    configs.append(BenchmarkConfig(
        model=base_model,
        dataset_name="random_medium",
        num_requests=100,
        arrival_rate=10.0,
        arrival_pattern=ArrivalPattern.POISSON,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        seed=42,
    ))
    
    # Configuration 2: Tensor parallelism
    configs.append(BenchmarkConfig(
        model=base_model,
        dataset_name="random_medium",
        num_requests=100,
        arrival_rate=10.0,
        arrival_pattern=ArrivalPattern.POISSON,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        seed=42,
    ))
    
    return configs


def main():
    parser = argparse.ArgumentParser(description="Run vLLM performance benchmarks")
    parser.add_argument("--config", type=str, help="Path to benchmark configuration JSON")
    parser.add_argument("--model", type=str, help="Model name/path")
    parser.add_argument("--dataset", type=str, default="random_medium", help="Dataset name")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--arrival-rate", type=float, default=10.0, help="Request arrival rate (req/s)")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pipeline-parallel", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--visualize", action="store_true", default=True, help="Generate visualizations")
    parser.add_argument("--analyze", action="store_true", default=True, help="Analyze bottlenecks")
    
    args = parser.parse_args()
    
    # Initialize components
    dataset_manager = DatasetManager.create_default_datasets()
    benchmark_runner = BenchmarkRunner(dataset_manager)
    plotter = BenchmarkPlotter(output_dir=f"{args.output_dir}/figures")
    bottleneck_analyzer = BottleneckAnalyzer()
    
    # Create benchmark configurations
    if args.config:
        # Load from JSON file
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        configs = [BenchmarkConfig(**c) for c in config_data]
    elif args.sweep:
        # Create sweep configurations
        configs = create_default_configs()
    else:
        # Single configuration from CLI args
        configs = [BenchmarkConfig(
            model=args.model or "meta-llama/Llama-2-7b-hf",
            dataset_name=args.dataset,
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            tensor_parallel_size=args.tensor_parallel,
            pipeline_parallel_size=args.pipeline_parallel,
            output_dir=args.output_dir,
        )]
    
    # Run benchmarks
    print(f"Running {len(configs)} benchmark configuration(s)...")
    all_results = benchmark_runner.run_sweep(configs)
    
    if not all_results:
        print("No benchmark results generated. Exiting.")
        return
    
    # Analyze bottlenecks
    if args.analyze:
        print("\n" + "="*60)
        print("BOTTLENECK ANALYSIS")
        print("="*60)
        
        for i, result in enumerate(all_results):
            print(f"\nConfiguration {i+1}:")
            analysis = bottleneck_analyzer.analyze(result)
            print(f"  Primary Bottleneck: {analysis.primary_bottleneck}")
            print(f"  Severity: {analysis.bottleneck_severity:.2f}")
            print(f"  Recommendations:")
            for rec in analysis.recommendations:
                print(f"    - {rec}")
    
    # Compare configurations
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("CONFIGURATION COMPARISON")
        print("="*60)
        
        comparison = bottleneck_analyzer.compare_configurations(all_results)
        print(f"\nBest Configuration Index: {comparison['best_config']}")
        print(f"Throughput Improvement: {comparison.get('throughput_improvement', 0):.1f}%")
        
        for config_info in comparison['configs']:
            print(f"\nConfig {config_info['index']}:")
            print(f"  Model: {config_info['model']}")
            print(f"  Tensor Parallel: {config_info['tensor_parallel']}")
            print(f"  Throughput: {config_info['throughput']:.2f} req/s")
            print(f"  Latency: {config_info['latency']:.3f} s")
    
    # Generate visualizations
    if args.visualize:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Generate plots for each result
        for i, result in enumerate(all_results):
            plotter.plot_latency_distribution(result, 
                save_path=f"{args.output_dir}/figures/latency_dist_{i}.png")
            plotter.plot_resource_usage(result,
                save_path=f"{args.output_dir}/figures/resource_usage_{i}.png")
        
        # Generate comparison plots
        if len(all_results) > 1:
            plotter.plot_throughput_comparison(all_results)
            plotter.plot_latency_vs_throughput(all_results)
            plotter.plot_phase_comparison(all_results)
        
        print(f"\nVisualizations saved to: {args.output_dir}/figures/")
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

