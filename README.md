# Ternary Walk — Random Walks on Z₃ State Space

**Ternary Walk** implements random walks on the ternary state space {-1, 0, +1}: simple symmetric walks, biased walks with probability weights, correlated walks with momentum, and Lévy flights with heavy-tailed jumps. It tracks occupation frequencies, return times, and state transition statistics.

## Why It Matters

Random walks are the stochastic foundation underlying many natural processes: diffusion, genetic drift, agent exploration, and stock price models. On Z₃, random walks have unique properties: the state space is bounded (only 3 states), so walks are recurrent — they return to each state infinitely often. The rate of mixing (how fast the walk approaches uniform distribution) determines how quickly an agent population loses its initial configuration. For fleet dynamics, ternary walks model how individual agent states drift over time, providing the baseline against which selection and coordination are measured.

## How It Works

### Simple Symmetric Walk

Each step: move to {-1, 0, +1} with equal probability ⅓, add to current position mod 3 (mapping back to {-1, 0, +1}):

```
delta = random_choice({-1, 0, +1}, uniform)
position = ((position + delta + 3) mod 3) - 1
```

O(1) per step. The stationary distribution is uniform: (⅓, ⅓, ⅓).

### Biased Walk

Probability weights (w_neg, w_zero, w_pos) determine the step distribution:

```
delta = -1 if r < w_neg/(w_neg+w_zero+w_pos)
        0  if r < (w_neg+w_zero)/total
        +1 otherwise
```

Bias toward +1 causes drift toward the positive state. Mixing time scales as O(1/bias²).

### Correlated Walk

Maintains momentum: the last step direction biases the next step. With momentum m:

```
w(last_direction) = 1 + m
w(opposite) = 1 - m/2
w(perpendicular) = 1
```

High momentum (m→1) creates long runs of the same state; low momentum (m→0) reduces to simple walk. O(1) per step.

### Lévy Flight

With probability p, make a large jump (flip to opposite state). With probability 1-p, make a simple walk step. The heavy tail creates intermittent bursts of exploration:

```
if random() < p: position = -position  (full flip)
else: simple_walk_step()
```

O(1) per step. Lévy flights mix faster than simple walks in certain topologies.

### Statistics

- **Occupation**: Frequency of visiting each state over the walk history. O(n) to compute.
- **Return time**: Steps since last visit to current state. O(n) lookup.
- **Step count**: Total steps taken.

## Quick Start

```rust
use ternary_walk::TernaryWalk;

let mut walk = TernaryWalk::new(0); // start at 0

// Simple walk
for _ in 0..1000 {
    walk.step_simple(0.5); // r = 0.5
}

let occupation = walk.occupation();
println!("State +1 visited {:.1}% of the time", occupation[&1] * 100.0);

// Biased walk
let mut biased = TernaryWalk::new(0);
biased.step_biased(0.7, 1.0, 2.0, 3.0); // biased toward +1
```

```bash
cargo add ternary-walk
```

## API

| Type / Function | Description |
|---|---|
| `TernaryWalk` | `{ position, history, step_count }` |
| `step_simple(r)` | Uniform random step |
| `step_biased(r, w_neg, w_zero, w_pos)` | Weighted random step |
| `step_correlated(r, momentum)` | Momentum-preserving step |
| `step_levy(r, heavy_tail_prob)` | Lévy flight step |
| `occupation() → HashMap<i8, f64>` | State visit frequencies |
| `return_time()` | Steps since last visit to current state |

## Architecture Notes

Random walks model stochastic agent dynamics in **SuperInstance**. The walk type determines the exploration strategy: simple walks are neutral (pure drift), biased walks model preference, correlated walks model habit, Lévy flights model disruptive exploration. The γ + η = C conservation manifests in the occupation frequencies: uniform occupation (balanced γ/η) is the maximum-entropy stationary state. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References:

- Pearson, Karl. "The Problem of the Random Walk," *Nature*, 72, 1905.
| Klafter, Joseph et al. *First Steps in Random Walks*, Oxford UP, 2011 — Lévy flights.
| Metropolis, Nicholas et al. "Equation of State Calculations," *J. Chem. Phys.*, 21(6), 1953 — Monte Carlo methods.

## License

MIT
