"""Agent module for trajectory collection.

This module provides:
- MiniSWERunner: Wrapper for mini-swe-agent with vLLM integration
- TrajectoryParser: Parse trajectories to extract (G, X, Q, T) tuples
"""

from src.agent.miniswe_runner import MiniSWERunner, RunnerConfig, create_runner_config_from_config
from src.agent.trajectory_parser import Trajectory, TrajectoryStep, parse_trajectory

__all__ = [
    # Trajectory parsing
    "Trajectory",
    "TrajectoryStep",
    "parse_trajectory",
    # Runner
    "MiniSWERunner",
    "RunnerConfig",
    "create_runner_config_from_config",
]
