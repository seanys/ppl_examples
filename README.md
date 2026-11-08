# PPL Examples

Examples of prescriptive policy learning (PPL), the method of *Teaching Agents the "Principle", Not the "Reward": Prescriptive Policy Learning*. PPL directly teaches agents principles of behavior, such as how to maintain balance, how to accumulate energy, and how to land smoothly, and agents learn their policies from these principles.

The website shows, for each scenario, the trajectories that the agent learns and the principles and references it is taught. Each scenario comes with a training script that runs on your own machine.

## Repository layout

- `docs/` the website, published with GitHub Pages
- `docs/examples/` training scripts, one folder per kind of system and one file per scenario; the website links to them for download
- `docs/data/` recorded episodes of the trained policies, played back on the website
- `runs/` created when you run a script; holds its evaluation results and models

## Running an example

```
pip install -r docs/examples/<system>/requirements.txt
python docs/examples/<system>/<scenario>.py
```

Each script prints its options with `--help`.
