# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This script demonstrates how the Python example service without needing
to use the bazel build system. Usage:

    $ python example_compiler_gym_service/demo_without_bazel.py

It is equivalent in behavior to the demo.py script in this directory.
"""
import logging
from pathlib import Path
from typing import Iterable
import os 
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


import my_env


# # Register the environment for use with gym.make(...).
# register(
#     id="example-v0",
#     entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
#     kwargs={
#         "service": Path(f"{os.path.dirname(os.path.realpath(__file__))}/example_service.py"),
#         "rewards": [RuntimeReward()],
#         "datasets": [ExampleDataset()],
#     },
# )


# def main():
#     # Use debug verbosity to print out extra logging information.
#     init_logging(level=logging.DEBUG)

#     # Create the environment using the regular gym.make(...) interface.
#     with gym.make("example-v0") as env:
#         env.reset()
#         for _ in range(20):
#             observation, reward, done, info = env.step(env.action_space.sample())
#             if done:
#                 env.reset()


# if __name__ == "__main__":
#     main()



def make_env():
    """Make the reinforcement learning environment for this experiment."""
    
    env = my_env.make(
        "example-v0",
        observation_space="features",
        reward_space="runtime",
    )
    # env = compiler_gym.make("loop_tool_env-v0")
    env = TimeLimit(env, max_episode_steps=20) # <<<< Must be here
    return env


from itertools import islice

if __name__ == "__main__":
    init_logging(level=logging.DEBUG)

    # ip_head and redis_passwords are set by ray cluster shell scripts
    ray_address = os.environ["RAY_ADDRESS"] if "RAY_ADDRESS" in os.environ else "auto"
    head_node_ip = os.environ["HEAD_NODE_IP"] if "HEAD_NODE_IP" in os.environ else "127.0.0.1"
    redis_password = os.environ["REDIS_PASSWORD"] if "REDIS_PASSWORD" in os.environ else "5241590000000000"
    print(ray_address, head_node_ip, redis_password)

    print("--- 2")
    ray.init(address=ray_address, _node_ip_address=head_node_ip, _redis_password=redis_password)
    # ray.init(local_mode=False, ignore_reinit_error=True)
    # sched = ASHAScheduler(metric="mean_accuracy", mode="max")



    with make_env() as env:
        train_benchmarks = list(islice(env.datasets['benchmark://example-v0'].benchmarks(), 1))

    def make_training_env(*args):
        """Make a reinforcement learning environment that cycles over the
        set of training benchmarks in use.
        """
        del args  # Unused env_config argument passed by ray
        return CycleOverBenchmarks(make_env(), train_benchmarks)


    tune.register_env("example-v0", make_training_env)
    print("--- 3")

    analysis = tune.run(
        PPOTrainer,
        checkpoint_at_end=True,
        stop={
            "episodes_total": 500,
            "training_iteration": 1
        },
        config={
            "seed": 0xCC,
            "num_workers": 1,
            # Specify the environment to use, where "compiler_gym" is the name we 
            # passed to tune.register_env().
            "env": "example-v0",
            # Reduce the size of the batch/trajectory lengths to match our short 
            # training run.
            "rollout_fragment_length": 5,
            "train_batch_size": 5,
            "sgd_minibatch_size": 5,
        },
        callbacks=[ WandbLoggerCallback(
                                project="loop_tool_agent",
                                api_key_file=str(LOOP_TOOL_ROOT) + "/wandb_key.txt",
                                log_config=False)]
    )
                        
    
    print("Best config is:", analysis.get_best_config(metric="mean_accuracy", mode="max"))
