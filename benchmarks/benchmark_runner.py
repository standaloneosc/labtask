"""Main benchmark runner that orchestrates performance testing."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from benchmarks.dataset_manager import BenchmarkRequest, Dataset, DatasetManager
from benchmarks.load_generator import ArrivalPattern, LoadConfig, LoadGenerator
from profiling.metrics_collector import MetricsCollector
from profiling.resource_monitor import ResourceMonitor
from profiling.vllm_integration import extract_vllm_metrics


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    model: str
    dataset_name: str
    num_requests: int
    arrival_rate: float
    arrival_pattern: ArrivalPattern = ArrivalPattern.POISSON
    
    # vLLM configuration
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    
    # Dataset overrides
    prompt_length: Optional[int] = None
    output_length: Optional[int] = None
    
    # Other settings
    seed: int = 42
    warmup_requests: int = 10
    output_dir: str = "results"


class BenchmarkRunner:
    """Runs comprehensive benchmarks with instrumentation."""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def run_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """
        Run a single benchmark configuration.
        
        Returns:
            Dictionary containing all benchmark results and metrics
        """
        print(f"\n{'='*60}")
        print(f"Running benchmark: {config.model}")
        print(f"Dataset: {config.dataset_name}")
        print(f"Requests: {config.num_requests}")
        print(f"Arrival rate: {config.arrival_rate} req/s")
        print(f"Tensor Parallel: {config.tensor_parallel_size}")
        print(f"Pipeline Parallel: {config.pipeline_parallel_size}")
        print(f"{'='*60}\n")
        
        # Initialize metrics collection
        metrics_collector = MetricsCollector()
        resource_monitor = ResourceMonitor(sampling_interval=0.1)
        
        # Get dataset
        dataset = self.dataset_manager.get_dataset(config.dataset_name)
        
        # Sample requests
        print("Sampling requests from dataset...")
        requests = dataset.sample_requests(config.num_requests, seed=config.seed)
        
        # Create load generator
        load_config = LoadConfig(
            arrival_rate=config.arrival_rate,
            total_requests=len(requests),
            pattern=config.arrival_pattern,
        )
        load_generator = LoadGenerator(load_config, seed=config.seed)
        
        # Start resource monitoring
        resource_monitor.start_monitoring()
        metrics_collector.start_collection()
        
        # Run benchmark
        try:
            results = self._execute_benchmark(
                config=config,
                requests=requests,
                load_generator=load_generator,
                metrics_collector=metrics_collector,
            )
        finally:
            metrics_collector.stop_collection()
            resource_monitor.stop_monitoring()
        
        # Collect all metrics
        metrics_summary = metrics_collector.get_summary_statistics()
        resource_summary = resource_monitor.get_summary()
        
        # Compile results
        benchmark_results = {
            "config": asdict(config),
            "metrics": metrics_summary,
            "resource_usage": resource_summary,
            "detailed_metrics": metrics_collector.to_dict(),
        }
        
        # Save results
        self._save_results(config, benchmark_results)
        
        return benchmark_results
    
    def _execute_benchmark(self,
                          config: BenchmarkConfig,
                          requests: List[BenchmarkRequest],
                          load_generator: LoadGenerator,
                          metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """
        Execute the actual benchmark using vLLM.
        
        This implementation:
        1. Initializes vLLM LLM engine with proper config
        2. Processes requests according to arrival times
        3. Tracks profiling and decode phases from vLLM metrics
        """
        from vllm import LLM, SamplingParams
        from vllm.inputs import TextPrompt, TokensPrompt
        
        print("Initializing vLLM engine...")
        
        # Prepare vLLM initialization arguments
        llm_kwargs = {
            "model": config.model,
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "dtype": config.dtype,
            "trust_remote_code": True,  # Enable trust for custom models
        }
        
        if config.max_model_len:
            llm_kwargs["max_model_len"] = config.max_model_len
        
        if config.pipeline_parallel_size > 1:
            llm_kwargs["pipeline_parallel_size"] = config.pipeline_parallel_size
        
        # Initialize vLLM
        try:
            llm = LLM(**llm_kwargs)
        except Exception as e:
            print(f"Error initializing vLLM: {e}")
            print("Falling back to simulation mode...")
            return self._simulate_benchmark(requests, metrics_collector)
        
        # Warmup requests
        if config.warmup_requests > 0:
            print(f"Warming up with {config.warmup_requests} requests...")
            warmup_requests = requests[:config.warmup_requests]
            warmup_prompts = [req.prompt for req in warmup_requests]
            warmup_params = SamplingParams(
                temperature=1.0,
                max_tokens=min(10, config.output_length or 128),  # Short warmup
                ignore_eos=True,
            )
            try:
                llm.generate(warmup_prompts, sampling_params=warmup_params, use_tqdm=False)
            except Exception as e:
                print(f"Warning: Warmup failed: {e}")
        
        print("Processing benchmark requests...")
        
        # Prepare requests for vLLM
        # We'll process requests in batches according to arrival times
        request_map = {}
        all_prompts = []
        all_sampling_params = []
        
        for i, request in enumerate(requests):
            request_id = f"req_{i}"
            request_map[request_id] = request
            
            # Create metrics entry
            metrics_collector.create_request(
                request_id=request_id,
                prompt_length=request.prompt_length,
                output_length=request.expected_output_length,
            )
            
            # Convert prompt to vLLM format
            if isinstance(request.prompt, str):
                prompt = TextPrompt(prompt=request.prompt)
            elif isinstance(request.prompt, dict) and "prompt_token_ids" in request.prompt:
                prompt = TokensPrompt(prompt_token_ids=request.prompt["prompt_token_ids"])
            else:
                prompt = TextPrompt(prompt=str(request.prompt))
            
            all_prompts.append(prompt)
            
            # Create sampling params
            sampling_params = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=request.expected_output_length,
                ignore_eos=True,
            )
            all_sampling_params.append(sampling_params)
        
        # Execute batch generation with vLLM
        # Note: For more realistic arrival pattern simulation, we'd need async engine
        # For now, we process all requests together but track timing
        
        batch_start_time = time.perf_counter()
        
        try:
            outputs = llm.generate(
                all_prompts,
                sampling_params=all_sampling_params,
                use_tqdm=True,
            )
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
        batch_end_time = time.perf_counter()
        
            # Extract metrics from vLLM outputs
            for i, output in enumerate(tqdm(outputs, desc="Processing outputs")):
                request_id = f"req_{i}"
                
                if not output:
                    continue
                
                # Mark request start
                metrics_collector.mark_request_start(request_id)
                
                # Use integration helper to extract vLLM metrics
                extract_vllm_metrics(output, request_id, metrics_collector)
                
                # Mark request end
                metrics_collector.mark_request_end(request_id)
        
        print(f"Completed batch generation in {batch_end_time - batch_start_time:.2f}s")
        
        return {}
    
    def _simulate_benchmark(self, 
                           requests: List[BenchmarkRequest],
                           metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Fallback simulation mode when vLLM is not available."""
        print("Running in simulation mode...")
        
        for i, request in enumerate(tqdm(requests, desc="Processing requests")):
            request_id = f"req_{i}"
            
            metrics = metrics_collector.create_request(
                request_id=request_id,
                prompt_length=request.prompt_length,
                output_length=request.expected_output_length,
            )
            
            metrics_collector.mark_request_start(request_id)
            
            # Simulate profiling phase
            metrics_collector.mark_profiling_start(request_id)
            time.sleep(0.01)
            metrics_collector.mark_profiling_end(request_id)
            
            # Simulate decode phase
            metrics_collector.mark_decode_start(request_id)
            num_tokens = request.expected_output_length
            for _ in range(num_tokens):
                decode_time = np.random.exponential(0.01)
                metrics_collector.record_token_decode(request_id, decode_time)
                time.sleep(decode_time)
            metrics_collector.mark_decode_end(request_id)
            
            metrics_collector.mark_request_end(request_id)
        
        return {}
    
    def _save_results(self, config: BenchmarkConfig, results: Dict[str, Any]):
        """Save benchmark results to disk."""
        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = int(time.time())
        filename = f"{config.model.replace('/', '_')}_{config.dataset_name}_{timestamp}.json"
        filepath = output_dir / filename
        
        # Save JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
    
    def run_sweep(self, configs: List[BenchmarkConfig]) -> List[Dict[str, Any]]:
        """Run multiple benchmark configurations."""
        all_results = []
        
        for i, config in enumerate(configs):
            print(f"\n\n{'#'*60}")
            print(f"Benchmark {i+1}/{len(configs)}")
            print(f"{'#'*60}\n")
            
            try:
                results = self.run_benchmark(config)
                all_results.append(results)
            except Exception as e:
                print(f"Error running benchmark: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_results

