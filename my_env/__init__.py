from pathlib import Path
import os
from compiler_gym.util.registration import register

import my_env.example_service


def register_env():
    print('\n\nyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy Register\n\n')

    register(
        id="example-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": Path(f"{os.path.dirname(os.path.realpath(__file__))}/example_service.py"),
            "rewards": [ my_env.example_service.RuntimeReward() ],
            "datasets": [ my_env.example_service.ExampleDataset() ],
        },
    )


def make(id: str, **kwargs):
    """Equivalent to :code:`compiler_gym.make()`."""
    register_env()
    import compiler_gym
    return compiler_gym.make(id, **kwargs)
