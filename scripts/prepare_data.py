#!/usr/bin/env python3
"""Prepare raw rollout data for SFT or offline RL training from trajectories."""

import argparse
import asyncio
import logging

from src.compression.data_preparation import DataPreparationPipeline
from src.config.config import Config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare SFT training data")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--compression-max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers for compression sampling.",
    )
    parser.add_argument(
        "--inference-max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for action inference.",
    )
    parser.add_argument(
        "--step-max-workers",
        type=int,
        default=1,
        help="Maximum number of trajectory steps to process concurrently.",
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    logger.info(f"Compression workers: {args.compression_max_workers}")
    logger.info(f"Inference workers: {args.inference_max_workers}")
    logger.info(f"Step workers: {args.step_max_workers}")

    pipeline = DataPreparationPipeline(
        config,
        compression_max_workers=args.compression_max_workers,
        inference_max_workers=args.inference_max_workers,
        step_max_workers=args.step_max_workers,
    )

    total_examples = await pipeline.prepare_data()
    logger.info(f"Data preparation complete! Total: {total_examples} examples")


if __name__ == "__main__":
    asyncio.run(main())
