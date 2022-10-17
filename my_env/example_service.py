#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
import logging
from pathlib import Path
from typing import Optional, Tuple

from compiler_gym.service import CompilationSession
from compiler_gym.service.proto import (
    ActionSpace,
    Benchmark,
    DoubleRange,
    Event,
    Int64Box,
    Int64Range,
    Int64Tensor,
    NamedDiscreteSpace,
    ObservationSpace,
    Space,
    StringSpace,
)
from compiler_gym.service.runtime import create_and_run_compiler_gym_service
import compiler_gym

import gym

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.wrappers import CycleOverBenchmarks
from compiler_gym.spaces import Reward
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register
from compiler_gym.wrappers import ConstrainedCommandline, TimeLimit


import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.rllib.agents.ppo import PPOTrainer

from loop_tool_service.paths import LOOP_TOOL_ROOT



class RuntimeReward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            name="runtime",
            observation_spaces=["runtime"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.previous_runtime = None

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.previous_runtime = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.previous_runtime is None:
            self.previous_runtime = observations[0]

        reward = float(self.previous_runtime - observations[0])
        self.previous_runtime = observations[0]
        return reward


class ExampleDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://example-v0",
            license="MIT",
            description="An example dataset",
        )
        self._benchmarks = {
            "/foo": Benchmark.from_file_contents(
                "benchmark://example-v0/foo", "Ir data".encode("utf-8")
            ),
            "/bar": Benchmark.from_file_contents(
                "benchmark://example-v0/bar", "Ir data".encode("utf-8")
            ),
        }

    def benchmark_uris(self):
        yield from (f"benchmark://example-v0{k}" for k in self._benchmarks.keys())

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        if uri.path in self._benchmarks:
            return self._benchmarks[uri.path]
        else:
            raise LookupError("Unknown program name")


class ExampleCompilationSession(CompilationSession):
    """Represents an instance of an interactive compilation session."""

    compiler_version: str = "1.0.0"

    # The action spaces supported by this service. Here we will implement a
    # single action space, called "default", that represents a command line with
    # three options: "a", "b", and "c".
    action_spaces = [
        ActionSpace(
            name="default",
            space=Space(
                named_discrete=NamedDiscreteSpace(
                    name=[
                        "a",
                        "b",
                        "c",
                    ],
                ),
            ),
        )
    ]

    # A list of observation spaces supported by this service. Each of these
    # ObservationSpace protos describes an observation space.
    observation_spaces = [
        ObservationSpace(
            name="ir",
            space=Space(
                string_value=StringSpace(length_range=Int64Range(min=0)),
            ),
            deterministic=True,
            platform_dependent=False,
            default_observation=Event(string_value=""),
        ),
        ObservationSpace(
            name="features",
            space=Space(
                int64_box=Int64Box(
                    low=Int64Tensor(shape=[3], value=[-100, -100, -100]),
                    high=Int64Tensor(shape=[3], value=[100, 100, 100]),
                ),
            ),
        ),
        ObservationSpace(
            name="runtime",
            space=Space(
                double_value=DoubleRange(min=0),
            ),
            deterministic=False,
            platform_dependent=True,
            default_observation=Event(
                double_value=0,
            ),
        ),
    ]

    def __init__(
        self, working_directory: Path, action_space: ActionSpace, benchmark: Benchmark
    ):
        super().__init__(working_directory, action_space, benchmark)
        logging.info("Started a compilation session for %s", benchmark.uri)

    def apply_action(self, action: Event) -> Tuple[bool, Optional[ActionSpace], bool]:
        num_choices = len(self.action_spaces[0].space.named_discrete.name)

        # This is the index into the action space's values ("a", "b", "c") that
        # the user selected, e.g. 0 -> "a", 1 -> "b", 2 -> "c".
        choice_index = action.int64_value
        logging.info("Applying action %d", choice_index)

        if choice_index < 0 or choice_index >= num_choices:
            raise ValueError("Out-of-range")

        # Here is where we would run the actual action to update the environment's
        # state.

        return False, None, False

    def get_observation(self, observation_space: ObservationSpace) -> Event:
        logging.info("Computing observation from space %s", observation_space)
        if observation_space.name == "ir":
            return Event(string_value="Hello, world!")
        elif observation_space.name == "features":
            observation = Event(int64_tensor=Int64Tensor(shape=[3], value=[0, 0, 0]))
            return observation
        elif observation_space.name == "runtime":
            return Event(double_value=0)
        else:
            raise KeyError(observation_space.name)


if __name__ == "__main__":
    create_and_run_compiler_gym_service(ExampleCompilationSession)
