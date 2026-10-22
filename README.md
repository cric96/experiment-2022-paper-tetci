# Learning when to run aggregate programs

Experiments for the TETCI paper on **learning the scheduling of aggregate-computing rounds**: every node of a
network runs a self-stabilizing [ScaFi](https://scafi.github.io/) program on the
[Alchemist](https://alchemistsimulator.github.io/) simulator and, after each round, a policy shared by all the nodes
decides how long the node sleeps before the next one. Waking up often makes the network converge quickly after a
change; sleeping saves energy. The policies are learned with tabular, offline deep and graph-based reinforcement
learning, and compared with fixed periods and a hand-crafted heuristic.

## The experiment

- **Network.** 100 nodes on a randomly perturbed 10 × 10 grid of 500 m × 500 m, communicating within 75 m.
- **Programs.** The nodes compute a *gradient* (distance from a source). The source moves to a far node every 100 s,
  so the network periodically has to re-converge. The generalization experiment also uses the *hop-count gradient*,
  *broadcast* and *collect* building blocks.
- **Episodes.** A run lasts 20 episodes of 200 s; every episode starts on a new deployment. Policies learn during the
  first 10 episodes (with decaying exploration) and are only exploited afterwards.
- **Actions.** The next wake-up period: one of 1–5 s (discrete learners) or any value in [0.5, 5] s (continuous ones).
- **Objective.** A trade-off weight β ∈ {0.5, 0.8, 0.95, 0.99, 1} balances convergence and energy. With the
  normalized period u ∈ [0, 1], a round costs β · I(s) · u² + (1 − β) · (1 − u)², where the *instability intensity*
  I(s) ∈ [0, 1] tells how much the node and its neighbourhood are changing (see `TradeOff` and `InstabilityIntensity`).
- **Metrics.** Rounds per episode (energy proxy), error integrated over time against the centrally computed ground
  truth, time spent with an unstable output, and reward.

| Policy | Learner | Simulation file | Gradle task |
| --- | --- | --- | --- |
| Fixed periods (1–5 s) | `SameActionAgent` | `simulationBaseline.yml` | `runSimulationBaseline` |
| Tabular Q-learning | `QLearning` | `simulation.yml` | `runSimulation` |
| Discrete BCQ (offline) | `OfflineLearner` | `simulationDeepOffline.yml` | `runSimulationDeepOffline` |
| Continuous actor-critic (offline) | `OfflineLearnerContinuous` | `simulationDeepOfflineContinous.yml` | `runSimulationDeepOfflineContinous` |
| EdgeGAT graph policy (offline) | `OfflineLearnerGNN` | `simulationDeepOfflineGNN.yml` | `runSimulationDeepOfflineGNN` |
| All of the above + heuristic, on unseen programs | `algoId` 0–9 | `simulationGeneralization.yml` | `runSimulationGeneralization` |

The offline learners are β-conditioned: every logged trajectory is relabelled for each value of β, so a single
policy serves every trade-off. In the generalization experiment the policies trained on the gradient are evaluated,
without retraining, on the other building blocks.

## Running the simulations

Requirements: a JDK ≥ 17 to run Gradle; the JDK used by the simulations is downloaded automatically, and so is
PyTorch (by DJL, on first use).

```bash
./gradlew runSimulationDeepOfflineGNN   # one experiment
./gradlew runAll                        # every experiment (hours)
```

`gradle.properties` controls the runs: with `batch=true` the simulations are headless and sweep the variables listed
in `variables` (seeds `random` 1–4, weights β, fixed `actions`); with `batch=false` a single run opens the graphical
interface with the effects in `effects/`. The exports are written to `build/exports/` (`*/episode` once per episode,
`*/full` every 2 s; `embeddings/` and `spatial/` hold the inspections of the run with seed 2).

## Charts, tables and statistics

`plotter.py` turns the exports into the paper figures (vector PDF + PNG), LaTeX `booktabs` tables with CSV copies,
and a generated `summary.md` with the key numbers:

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python plotter.py --exports build/exports --out charts     # add --strict to fail on any warning
```

The code lives in the `analysis` package: `io` (Alchemist CSV parser), `data`/`dataset` (tidy tables), `stats` and
`pareto` (confidence intervals, significance tests, frontier comparisons), `results`, `style` (IEEE paper style and
the per-method colours and markers), `figures/*` (one module per figure family), `tables` and `report`.

## Development

```bash
./gradlew spotlessApply test        # format and run the Scala tests
.venv/bin/python -m pytest          # tests of the analysis
```

The Scala tests include golden masters that run every simulation for a few episodes and compare the state of the
nodes with pinned values, so refactorings can be proven not to change the experiments. `CLAUDE.md` describes the
architecture, the style and the workflow of the code base. The CI runs the formatting check, the Scala tests and the
analysis tests on every push.
