"""Load generator for simulating different request arrival patterns."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, List, Optional

import numpy as np


class ArrivalPattern(Enum):
    """Types of arrival patterns."""
    CONSTANT = "constant"  # Constant arrival rate
    POISSON = "poisson"    # Poisson process
    BURST = "burst"        # Burst arrival pattern
    SINUSOIDAL = "sinusoidal"  # Sinusoidal variation


@dataclass
class LoadConfig:
    """Configuration for load generation."""
    arrival_rate: float  # Requests per second
    total_requests: int
    pattern: ArrivalPattern = ArrivalPattern.POISSON
    burst_size: Optional[int] = None  # For burst pattern
    burst_interval: Optional[float] = None  # For burst pattern
    duration: Optional[float] = None  # Optional: duration in seconds


class LoadGenerator:
    """Generates request arrival times according to different patterns."""
    
    def __init__(self, config: LoadConfig, seed: Optional[int] = None):
        """
        Initialize load generator.
        
        Args:
            config: Load configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def generate_arrival_times(self) -> List[float]:
        """Generate arrival times based on configured pattern."""
        if self.config.pattern == ArrivalPattern.CONSTANT:
            return self._generate_constant()
        elif self.config.pattern == ArrivalPattern.POISSON:
            return self._generate_poisson()
        elif self.config.pattern == ArrivalPattern.BURST:
            return self._generate_burst()
        elif self.config.pattern == ArrivalPattern.SINUSOIDAL:
            return self._generate_sinusoidal()
        else:
            raise ValueError(f"Unknown arrival pattern: {self.config.pattern}")
    
    def _generate_constant(self) -> List[float]:
        """Generate constant arrival rate."""
        interval = 1.0 / self.config.arrival_rate if self.config.arrival_rate > 0 else 1.0
        times = []
        current_time = 0.0
        
        for _ in range(self.config.total_requests):
            times.append(current_time)
            current_time += interval
        
        return times
    
    def _generate_poisson(self) -> List[float]:
        """Generate Poisson arrival process."""
        if self.config.arrival_rate <= 0:
            return []
        
        times = []
        current_time = 0.0
        
        for _ in range(self.config.total_requests):
            # Exponential inter-arrival time for Poisson process
            inter_arrival = self.rng.exponential(1.0 / self.config.arrival_rate)
            current_time += inter_arrival
            times.append(current_time)
        
        return times
    
    def _generate_burst(self) -> List[float]:
        """Generate burst arrival pattern."""
        if self.config.burst_size is None or self.config.burst_interval is None:
            raise ValueError("burst_size and burst_interval required for burst pattern")
        
        times = []
        current_time = 0.0
        requests_remaining = self.config.total_requests
        
        while requests_remaining > 0:
            # Generate burst
            burst_count = min(self.config.burst_size, requests_remaining)
            burst_interval = 1.0 / (self.config.arrival_rate * self.config.burst_size)
            
            for _ in range(burst_count):
                # Small jitter within burst
                jitter = self.rng.uniform(0, burst_interval * 0.1)
                times.append(current_time + jitter)
                current_time += burst_interval
                requests_remaining -= 1
            
            # Wait before next burst
            if requests_remaining > 0:
                current_time += self.config.burst_interval - (burst_interval * burst_count)
        
        return sorted(times)
    
    def _generate_sinusoidal(self) -> List[float]:
        """Generate sinusoidal arrival pattern."""
        if self.config.duration is None:
            # Estimate duration based on arrival rate
            duration = self.config.total_requests / self.config.arrival_rate
        else:
            duration = self.config.duration
        
        times = []
        period = duration / 2  # Two periods over duration
        
        for i in range(self.config.total_requests):
            # Normalize to [0, 1]
            t_norm = i / self.config.total_requests
            
            # Sinusoidal variation in arrival rate
            rate_multiplier = 1.0 + 0.5 * np.sin(2 * np.pi * t_norm * 2)
            current_rate = self.config.arrival_rate * rate_multiplier
            
            # Generate next arrival
            if i == 0:
                inter_arrival = 1.0 / current_rate if current_rate > 0 else 1.0
                current_time = inter_arrival
            else:
                inter_arrival = self.rng.exponential(1.0 / current_rate) if current_rate > 0 else 1.0
                current_time = times[-1] + inter_arrival
            
            times.append(current_time)
        
        return times
    
    def schedule_requests(self, requests: List, start_time: float = None) -> Iterator[tuple]:
        """
        Schedule requests according to arrival pattern.
        
        Yields (request, arrival_time) tuples.
        """
        if start_time is None:
            start_time = time.perf_counter()
        
        arrival_times = self.generate_arrival_times()
        
        # Sort by arrival time
        scheduled = list(zip(requests, arrival_times))
        scheduled.sort(key=lambda x: x[1])
        
        for request, relative_time in scheduled:
            absolute_time = start_time + relative_time
            yield request, absolute_time
    
    def wait_until(self, target_time: float):
        """Wait until target time."""
        current_time = time.perf_counter()
        sleep_time = target_time - current_time
        if sleep_time > 0:
            time.sleep(sleep_time)

