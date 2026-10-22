# CLAUDE.md

Guidance for working on this repository: experiments for the TETCI paper on
**learning when to execute the rounds of aggregate-computing programs** (ScaFi on Alchemist)
with tabular, offline-deep and graph-based reinforcement learning.

## Build, run, test

- Gradle 8.8 does **not** run on the system JDK 25. Use the JDK 17 already provisioned by Gradle:
  `export JAVA_HOME=~/.gradle/jdks/eclipse_adoptium-17-amd64-linux.2`
- Compile: `./gradlew compileScala`
- Tests: `./gradlew test` (ScalaTest, specs live in `src/test/scala`, mirroring the main packages)
- Format: `./gradlew spotlessApply` (scalafmt, configuration in `.scalafmt.conf`); `spotlessCheck` must pass.
- One Gradle task per simulation file: `src/main/yaml/simulationDeepOffline.yml` → `./gradlew runSimulationDeepOffline`.
  `gradle.properties` sets `batch=true` (headless) and the variables to sweep; exports land in `build/exports/`.
- Charts: `plotter.py` (Python, only for figures; see `requirements.txt`). Use `uv` — the system Python has no matplotlib.

## Architecture

| Package | Role |
| --- | --- |
| `it.unibo.scafi.Main` | Aggregate program executed by every node: computes the building block (gradient, hop gradient, broadcast, collect), builds the `AgentState`, computes the reward and actuates the next wake-up (`nextWakeUp` molecule). |
| `it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent` | Global reaction: trains the learner, closes episodes by re-deploying the nodes, computes ground truths, exports spatial snapshots. |
| `...timedistributions.NextWakeUp` | Time distribution that reads `nextWakeUp` to schedule the node's next round. |
| `it.unibo.learning.agents` | `Learner` implementations: `QLearning`, `OfflineLearner` (discrete BCQ), `OfflineLearnerContinuous` (TD3+BC-like), `OfflineLearnerGNN` (EdgeGAT actor-critic), `HeuristicAdaptiveAgent`, `SameActionAgent` (fixed-period baselines). |
| `it.unibo.learning.network` | State encoders (`NeuralNetworkRL`) and the DJL/PyTorch wrapper (`TorchNetwork`). |
| `it.unibo.alchemist.loader.export` | Custom Alchemist extractors used by the YAML `export` sections. |

**The YAML files are a public API.** They instantiate learners and value objects through embedded Scala snippets
(e.g. `new OfflineLearnerGNN(0.5, 5.0, random.as[Double], weight.as[Double])`) and refer to classes, molecules and
constructor parameters by name. These snippets are compiled only at simulation start-up, so the compiler will not
catch a broken signature: when changing a constructor, class name, package or molecule name, grep `src/main/yaml`
and update it in the same commit.

## How to change code here

- **Small, incremental, behaviour-preserving steps.** One concern per commit; the build and the tests are green after
  every commit.
- **TDD.** For new behaviour: failing test → minimal code → refactor. For refactoring untested code: first write a
  *characterization* test that pins the current behaviour (it must pass on the old code), then refactor while it
  stays green.
- **Reproducibility is behaviour.** Experiments are seeded: a refactoring must not change the sequence of calls on a
  `Random`/`RandomGenerator`, the order of floating-point operations in hot numerical code (fast MLP/GAT inference,
  rewards, encoders), nor the names/order of exported CSV columns that `plotter.py` reads. If a change intentionally
  alters results, say so in the commit message.
- Commit messages follow Conventional Commits (`refactor(agents): ...`, `test(network): ...`, `fix: ...`).
  **No AI attribution lines** (`Co-Authored-By`, "Generated with ...").

## Scala style (Scala 2.13)

### Values and state
- `val` and immutable collections by default. `var` is acceptable only for the inherently stateful Alchemist glue
  (reactions, learners that accumulate data); keep it `private` and as narrow as possible.
- Prefer expressions to statements: no `return`, no "init then mutate" — use `if`/`match`/`fold` as expressions.
- Model fixed sets of alternatives as ADTs (`sealed trait` + `case object`/`final case class`), not as `String` or
  `Int` codes. Parse external values (YAML, molecules) once, at the boundary, into the ADT.
- Use a `final case class` instead of a tuple as soon as it has more than two fields or crosses a module boundary.
- **No magic numbers.** Every literal with a domain meaning (thresholds, gains, sizes, times, seeds, file names,
  column names) is a named constant: `val UpperCamelCase` inside an `object` (the companion or a dedicated
  `object ...Constants`), with a one-line doc comment saying what it means. Only neutral literals are allowed inline
  (`0`, `1`, `0.0`, `1.0`, `2` in `x * x`-like arithmetic, empty collections). If a value is derived from others
  (e.g. the start of the last episode = `(episodes - 1) * episodeLength`), compute it instead of hard-coding it.

### Absence and failure
- No `null` (wrap Java values with `Option(...)`), no `Option.get`, no `head` on possibly empty collections:
  use `fold`, `getOrElse`, `collectFirst`, `headOption`, `maxByOption`, `minOption`.
- Do not swallow errors: never `catch { case _: Throwable => }`. Use `scala.util.Try`/`Either`, or
  `case NonFatal(e)` with a meaningful fallback.
- `asInstanceOf` only at the Alchemist/Java boundary (molecule values, YAML scripts), wrapped in a single helper.
- Resources (files, DJL managers) are handled with `scala.util.Using`.

### Functions and collections
- **Short methods.** A method does one thing and fits in a screen: aim for ≤ 15 lines of body, never more than ~25.
  When a method grows, extract well-named sub-methods (private, or local `def`s when they are tiny) until the
  top-level method reads as a sequence of steps. Deep nesting (`if` inside `foreach` inside `match`) is a signal to
  extract.
- **Short files, one domain concept each.** A file that grows past ~200 lines, or that mixes concerns (e.g. training,
  ground truth, CSV export and episode management in one reaction), is split into types named after the domain
  concepts they implement (`GroundTruth`, `EpisodeManager`, `SpatialSnapshotExporter`, `StratifiedTransitions`, ...),
  each in its own file, with the original class delegating to them.
- Small pure functions, placed in the companion `object` or in a dedicated `object`; keep side effects
  (Alchemist node access, file I/O, logging) at the edges so that the logic is testable without a simulator.
- Use combinators (`map`, `collect`, `foldLeft`, `sliding`, `zip`, `groupMapReduce`, `sum`, `maxBy`) instead of
  index loops and accumulators. `while` loops are allowed only in measured hot paths (neural-network inference
  executed every round); encapsulate them in one small, documented function.
- Don't repeat yourself across learners: shared behaviour goes in a trait or in a helper object, not copy-paste.
- Prefer `Iterator`/`LazyList` over manual buffers when building sequences on demand.

### Types and API
- Explicit return types on every public member.
- `final` for classes not meant to be extended; `private` for everything that is not API.
- Implicit enrichments: `implicit final class XOps(private val self: X) extends AnyVal`.
- Named and default arguments for constructors used from YAML; keep positional order stable.
- Traits provide sensible default implementations instead of forcing every implementor to write no-op overrides.

### Naming and layout
- **No compact or abbreviated names**: `groundTruth`, not `gtFin`; `ticks`, not `tks`; `maybeParent`/`parent`, not
  `opt`/`parentOpt`; `previous`/`current`, not `prev`/`curr`; `manager`, not `m`; `neighbour`, not `n`/`nId`.
  Single-letter names only for conventional math in a short expression (`u` for the normalized period of a formula
  that documents it, `i`/`k` for indices in a hot loop) and `_` placeholders in one-line lambdas.
- `camelCase` for terms, `UpperCamelCase` for types and constants.
- Imports at the top of the file; no fully qualified names inline (`new java.io.File`, `scala.collection.mutable.X`).
  Use `scala.jdk.CollectionConverters._` for Java interop.
- Formatting is owned by scalafmt (`maxColumn = 120`): don't hand-format, run `spotlessApply`.

### Tests
- `SimulationGoldenMasterSpec` runs every YAML for a few episodes (no exporters) and compares fingerprints of the
  node states with pinned values: it is the safety net for refactoring Main, CentralAgent and the learners.
  The pinned values depend on the PyTorch CPU kernels, so they may need re-pinning on a different machine.
- ScalaTest `AnyFlatSpec with Matchers`; one spec per production class, same package, named `<Class>Spec`.
- Build `AgentState` fixtures with the helpers in the test sources rather than ad-hoc maps in every test.
- Use fixed seeds; when pinning numerical behaviour, compare with a tolerance only if the code legitimately
  changes the arithmetic, otherwise compare exactly.
