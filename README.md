# PPL Examples

Examples of prescriptive policy learning (PPL), the method of *Teaching Agents the "Principle", Not the "Reward": Prescriptive Policy Learning*. PPL directly teaches agents principles of behavior, such as how to maintain balance, how to accumulate energy, and how to land smoothly, and agents learn from these principles without any reward.

The website shows, for each scenario, the trajectories that the agent learns and the principles and references it is taught. Each scenario comes with a training script that runs on your own machine.

## Repository layout

- `examples/` training scripts, one folder per kind of system and one file per scenario
- `docs/` the website, published with GitHub Pages
- `runs/` created when you run a script; holds its evaluation results and models

## Running an example

```
pip install -r examples/<system>/requirements.txt
python examples/<system>/<scenario>.py
```

Each script prints its options with `--help`.
