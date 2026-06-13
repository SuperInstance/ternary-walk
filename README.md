# ternary-walk

Ternary random walks on the state space **{-1, 0, +1}** — simple symmetric, biased, correlated (persistent), and Lévy flight variants. Includes ensemble statistics, occupation frequencies, return times, and autocorrelation analysis.

## Why It Matters

Random walks on finite state spaces are the backbone of stochastic agent models. The ternary state space {-1, 0, +1} is especially natural because it models agents with a neutral option — the vast majority of real-world decisions are not binary.

Different walk types model different agent behaviors:
- **Simple symmetric:** unbiased exploration
- **Biased:** agents with inherent drift (toward choose or avoid)
- **Correlated:** agents with momentum/inertia (persistent behavior)
- **Lévy flight:** agents with occasional large shifts (regime changes)

Understanding the stationary distributions and return times of these walks tells us when an agent population will reach equilibrium and how stable that equilibrium is.

## How It Works

### State Space and Transition Structure

The walk operates on **ℤ₃ = {-1, 0, +1}** with modular arithmetic. Position updates use:

```
new_position = ((position + delta) mod 3) mapped to {-1, 0, +1}
```

This wrapping ensures the walk stays on the ternary lattice.

### Simple Symmetric Walk

At each step, the delta Δ is drawn uniformly from {-1, 0, +1}:

```
P(Δ = -1) = P(Δ = 0) = P(Δ = +1) = 1/3
```

**Stationary distribution:** Uniform — each state has probability 1/3.

**Expected return time:** E[T_return] = 3 (for any state in ℤ₃).

- **Complexity:** O(1) per step

### Biased Walk

Custom weights (w_neg, w_zero, w_pos) control transition probabilities:

```
P(Δ = -1) = w_neg / W,  P(Δ = 0) = w_zero / W,  P(Δ = +1) = w_pos / W
```

Where W = w_neg + w_zero + w_pos.

**Stationary distribution:** Proportional to the transition weights (detailed balance on ℤ₃).

- **Complexity:** O(1) per step

### Correlated (Persistent) Walk

Uses momentum to bias the next step in the direction of the last step:

```
If last_delta = -1:  weights = (1 + p, 1, 1 − p/2)
If last_delta =  0:  weights = (1, 1 + p, 1)
If last_delta = +1:  weights = (1 − p/2, 1, 1 + p)
```

Where p ∈ [0, 1] is the momentum parameter. High p creates long runs in one direction.

- **Complexity:** O(1) per step
- **Autocorrelation:** Exponential decay with characteristic time τ ≈ 1/(1 − p)

### Lévy Flight

Heavy-tailed step distribution: with probability `q`, the walker makes a large jump (full state flip); otherwise, a normal symmetric step:

```
P(flip to -position) = q
P(normal step)        = 1 − q
```

This produces intermittent dynamics — long periods of small steps punctuated by sudden regime changes. Lévy flights are scale-free: the step variance is infinite in the continuous limit.

- **Complexity:** O(1) per step

### Ensemble Statistics

The `WalkEnsemble` aggregates N independent walkers:
- **Mean occupation:** ensemble-averaged state frequencies → stationary distribution estimate
- **Autocorrelation at lag k:** ρ(k) = E[X_t · X_{t+k}] — characterizes memory

## Quick Start

```rust
use ternary_walk::*;

// Simple symmetric walk
let mut walk = TernaryWalk::new(0);
for i in 0..1000 {
    walk.step_simple((i as f64 * 0.001) % 1.0);
}
let occ = walk.occupation();
println!("Occupation: {:?}", occ);

// Biased walk (drift toward +1)
let mut biased = TernaryWalk::new(0);
for i in 0..1000 {
    biased.step_biased((i as f64 * 0.001) % 1.0, 0.2, 0.3, 0.5);
}

// Correlated walk (momentum = 0.7)
let mut corr = TernaryWalk::new(0);
for i in 0..1000 {
    corr.step_correlated((i as f64 * 0.001) % 1.0, 0.7);
}

// Lévy flight (heavy tail probability = 0.05)
let mut levy = TernaryWalk::new(0);
for i in 0..1000 {
    levy.step_levy((i as f64 * 0.001) % 1.0, 0.05);
}

// Ensemble analysis
let ensemble = WalkEnsemble::new(100, 0);
println!("Autocorrelation at lag 5: {:.4}", ensemble.autocorrelation(5));
```

## API

| Type | Purpose |
|---|---|
| `TernaryWalk` | Single walker with step history and occupation stats |
| `WalkEnsemble` | Collection of walkers for ensemble averaging |
| `TernaryWalk::step_simple` | Symmetric step (uniform Δ) |
| `TernaryWalk::step_biased` | Weighted step (custom probabilities) |
| `TernaryWalk::step_correlated` | Persistent step (momentum-based) |
| `TernaryWalk::step_levy` | Lévy flight (occasional large jumps) |
| `TernaryWalk::occupation` | State frequency HashMap |
| `TernaryWalk::return_time` | Steps since last visit to current state |
| `WalkEnsemble::mean_occupation` | Ensemble-averaged stationary distribution |
| `WalkEnsemble::autocorrelation` | Temporal autocorrelation at given lag |

## Architecture Notes

The walk's stationary distribution connects to the **γ + η = C** conservation law. For a simple symmetric walk, the stationary distribution is (1/3, 1/3, 1/3) — meaning γ = η = neutrals = 1/3 at equilibrium, perfectly satisfying conservation. Biased walks shift this distribution: a drift toward +1 increases γ at the expense of η, but the sum γ + η always equals the non-neutral fraction.

The correlated walk's autocorrelation function ρ(k) = E[X_t · X_{t+k}] decays as exp(−k/τ), providing the timescale over which agent "memory" of past decisions persists. This is the temporal analogue of the conservation law's spatial constraint.

## References

- Feller, W. (1968). *An Introduction to Probability Theory and Its Applications, Vol. 1.* Wiley. — Random walks on finite groups.
- Klafter, J., Shlesinger, M. F. & Zumofen, G. (1996). *"Beyond Brownian Motion."* Physics Today. — Lévy flights.
- Weiss, G. H. (1994). *Aspects and Applications of the Random Walk.* North-Holland.

## License

MIT
