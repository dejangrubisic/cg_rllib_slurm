# cg_rllib_slurm
This is the reproducer for halting problem of CG on SLURM. To show the problem, we run
non CompilerGym TrainMNIST environment that successfully finish on slurm, while CompilerGym
halts.

To run TrainMNIST:
```
python launcher/slurm_launch.py --app=main.py
```

To run CompilerGym:
```
python launcher/slurm_launch.py --app=main_cg.py
```

