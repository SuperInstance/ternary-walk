# ternary-walk

**Random walks on the ternary domain {-1, 0, +1}. Where does chance take you?**

A random walk is the simplest stochastic process: start somewhere, take a random step, repeat. In ternary, each step adds -1, 0, or +1 to your current position (clamped to {-1, 0, +1}). The walk is confined to three states — it can't escape. This confinement makes ternary walks fundamentally different from unbounded walks: they're *ergodic* (every state is reachable from every other state) and *recurrent* (every state is visited infinitely often).

This crate implements multiple walk types: simple, biased, correlated, absorbing, reflecting, and spatial. Each captures a different aspect of randomness in ternary space.

## What's Inside

- **`SimpleTernaryWalk`** — uniform random steps in {-1, 0, +1}, clamped
- **`BiasedWalk`** — weighted probabilities for each direction
- **`CorrelatedWalk`** — momentum: each step depends on the previous direction
- **`AbsorbingWalk`** — stops when hitting a specific value
- **`ReflectingWalk`** — boundaries bounce the walker back
- **`SpatialTernaryWalk`** — walk on a 2D ternary grid, reading values at each step
- **`WalkStatistics`** — position history, min, max, mean, variance, zero crossings, time in each state
- **`FirstPassageTime`** — steps to first reach each value
- **`OccupationMeasure`** — fraction of time spent in each state
- **`ReturnTimeDistribution`** — steps between returns to each state

## Quick Example

```rust
use ternary_walk::*;

let mut walk = BiasedWalk::new([1, 2, 3]); // weights: [-1, 0, +1]
// +1 is 3x more likely than -1

let mut stats = WalkStatistics::new();
for _ in 0..1000 {
    let step = walk.step();
    stats.record(step);
}

println!("Mean: {:.2}", stats.mean());
println!("Occupation: -1={:.1}%, 0={:.1}%, +1={:.1}%",
    stats.fraction_at(-1) * 100.0,
    stats.fraction_at(0) * 100.0,
    stats.fraction_at(1) * 100.0);
```

## The Deeper Truth

**Ternary walks are always recurrent.** In an unbounded random walk (on the integers), the walk can drift to infinity and never return. On ternary {-1, 0, +1}, it can't — there are only three states, and the walk visits all of them infinitely often. This means every ternary walk has a stationary distribution, and the occupation measure converges to it regardless of the starting state.

The `no_std` implementation means this runs on bare metal — microcontrollers, embedded systems, anywhere you need random ternary dynamics without an operating system.

**Use cases:**
- **Stochastic modeling** — random exploration of ternary state spaces
- **Monte Carlo simulation** — ternary walks as random number generators
- **Noise generation** — random ternary sequences for dithering or testing
- **Agent behavior** — random exploration strategies
- **Markov chain analysis** — ternary walks are Markov chains with 3 states

## See Also

- **ternary-drift** — population-level random walks (Wright-Fisher)
- **ternary-markov** — Markov chain prediction (walks with memory)
- **ternary-life** — deterministic walks on grids (Game of Life)
- **ternary-fib** — deterministic cyclic walks (period 8)

## Install

```bash
cargo add ternary-walk
```

## License

MIT
