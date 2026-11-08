# Training scripts

Each folder holds one kind of system, and each file trains one scenario:

- `pendulums/` pendulums and linkages
- `thrust_vehicles/` thrust vehicles
- `ground_vehicles/` ground vehicles
- `legged/` legged jumping and locomotion
- `manipulation/` object manipulation

Every script is self-contained: it holds the PPL update and the principles of its scenario, and depends only on public libraries listed in the `requirements.txt` of its folder. Scenario parameters are set on the command line. While running, a script writes its evaluation results and models to `runs/<time>_<scenario>/` and resumes from there after an interruption.
