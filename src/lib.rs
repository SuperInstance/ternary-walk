//! Ternary random walks: simple, biased, correlated, and Lévy flights on Z₃ state space.

use std::collections::HashMap;

/// Ternary walk step generator
pub struct TernaryWalk {
    pub position: i8,
    pub history: Vec<i8>,
    step_count: usize,
}

impl TernaryWalk {
    pub fn new(start: i8) -> Self {
        assert!(start >= -1 && start <= 1);
        Self { position: start, history: vec![start], step_count: 0 }
    }

    /// Simple symmetric random walk on {-1, 0, 1}
    pub fn step_simple(&mut self, r: f64) -> i8 {
        let delta = if r < 1.0/3.0 { -1 } else if r < 2.0/3.0 { 0 } else { 1 };
        self.position = ((self.position as i32 + delta as i32 + 3) % 3 - 1) as i8;
        self.history.push(self.position);
        self.step_count += 1;
        self.position
    }

    /// Biased walk: probability weights for (-1, 0, +1)
    pub fn step_biased(&mut self, r: f64, w_neg: f64, w_zero: f64, w_pos: f64) -> i8 {
        let total = w_neg + w_zero + w_pos;
        let delta = if r < w_neg / total { -1 } else if r < (w_neg + w_zero) / total { 0 } else { 1 };
        self.position = ((self.position as i32 + delta as i32 + 3) % 3 - 1) as i8;
        self.history.push(self.position);
        self.step_count += 1;
        self.position
    }

    /// Correlated walk: tendency to continue in same direction
    pub fn step_correlated(&mut self, r: f64, momentum: f64) -> i8 {
        let last_delta = if self.history.len() >= 2 {
            let prev = self.history[self.history.len()-2];
            let cur = self.history[self.history.len()-1];
            ((cur as i32 - prev as i32 + 3) % 3 - 1) as i8
        } else { 0 };

        let (w_neg, w_zero, w_pos) = match last_delta {
            -1 => (1.0 + momentum, 1.0, 1.0 - momentum * 0.5),
            0 => (1.0, 1.0 + momentum, 1.0),
            1 => (1.0 - momentum * 0.5, 1.0, 1.0 + momentum),
            _ => (1.0, 1.0, 1.0),
        };
        self.step_biased(r, w_neg.max(0.1), w_zero, w_pos.max(0.1))
    }

    /// Lévy flight: occasional large jumps (full state flip)
    pub fn step_levy(&mut self, r: f64, heavy_tail_prob: f64) -> i8 {
        if r < heavy_tail_prob {
            // Large jump: flip to opposite
            self.position = -self.position;
        } else {
            self.step_simple(r / heavy_tail_prob);
        }
        self.history.push(self.position);
        self.step_count += 1;
        self.position
    }

    pub fn step_count(&self) -> usize { self.step_count }

    /// State occupation frequencies
    pub fn occupation(&self) -> HashMap<i8, f64> {
        let mut counts = HashMap::new();
        for &s in &self.history { *counts.entry(s).or_insert(0usize) += 1; }
        let total = self.history.len() as f64;
        counts.into_iter().map(|(k, v)| (k, v as f64 / total)).collect()
    }

    /// Return time: how many steps since last visit to current state
    pub fn return_time(&self) -> usize {
        let target = self.position;
        for i in (0..self.history.len().saturating_sub(1)).rev() {
            if self.history[i] == target {
                return self.history.len() - 1 - i;
            }
        }
        self.history.len()
    }
}

/// Ensemble of ternary walks for statistical analysis
pub struct WalkEnsemble {
    pub walks: Vec<TernaryWalk>,
}

impl WalkEnsemble {
    pub fn new(n: usize, start: i8) -> Self {
        Self { walks: (0..n).map(|_| TernaryWalk::new(start)).collect() }
    }

    /// Mean occupation across all walkers
    pub fn mean_occupation(&self) -> HashMap<i8, f64> {
        let mut result = HashMap::new();
        for w in &self.walks {
            for (&state, &freq) in &w.occupation() {
                *result.entry(state).or_insert(0.0) += freq / self.walks.len() as f64;
            }
        }
        result
    }

    /// Autocorrelation at given lag
    pub fn autocorrelation(&self, lag: usize) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for w in &self.walks {
            if lag >= w.history.len() { continue; }
            for i in 0..(w.history.len() - lag) {
                sum += w.history[i] as f64 * w.history[i + lag] as f64;
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_walk_bounds() {
        let mut w = TernaryWalk::new(0);
        for i in 0..100 {
            w.step_simple((i as f64 * 0.01) % 1.0);
            assert!(w.position >= -1 && w.position <= 1);
        }
    }

    #[test]
    fn test_occupation_approximately_uniform() {
        let mut w = TernaryWalk::new(0);
        for i in 0..10000 {
            w.step_simple(((i * 7 + 3) as f64 * 0.0001) % 1.0);
        }
        let occ = w.occupation();
        for state in [-1i8, 0, 1] {
            let freq = occ.get(&state).copied().unwrap_or(0.0);
            assert!(freq > 0.25 && freq < 0.42, "State {} freq {}", state, freq);
        }
    }

    #[test]
    fn test_biased_walk_toward_positive() {
        let mut w = TernaryWalk::new(0);
        for i in 0..10000 {
            w.step_biased(((i * 13) as f64 * 0.0001) % 1.0, 0.1, 0.2, 0.7);
        }
        let occ = w.occupation();
        let pos_freq = occ.get(&1).copied().unwrap_or(0.0);
        let neg_freq = occ.get(&-1).copied().unwrap_or(0.0);
        assert!(pos_freq > neg_freq, "Positive should dominate");
    }

    #[test]
    fn test_correlated_walk_persistence() {
        let mut w = TernaryWalk::new(0);
        w.step_simple(0.9); // go +1
        for i in 0..5000 {
            w.step_correlated(((i * 17) as f64 * 0.0001) % 1.0, 2.0);
        }
        // With high momentum, should show autocorrelation
        let ensemble = WalkEnsemble { walks: vec![TernaryWalk { position: w.position, history: w.history.clone(), step_count: w.step_count }] };
        let ac1 = ensemble.autocorrelation(1);
        assert!(ac1 > 0.0, "Correlated walk should have positive autocorrelation");
    }

    #[test]
    fn test_levy_flight_heavy_jumps() {
        let mut w = TernaryWalk::new(1);
        let mut flips = 0;
        for i in 0..1000 {
            let prev = w.position;
            w.step_levy(((i * 11) as f64 * 0.001) % 1.0, 0.3);
            if w.position == -prev { flips += 1; }
        }
        assert!(flips > 50, "Should have many flips with heavy tail");
    }

    #[test]
    fn test_return_time_nonzero() {
        let mut w = TernaryWalk::new(0);
        for i in 0..100 {
            w.step_simple(((i * 7 + 3) as f64 * 0.01) % 1.0);
        }
        let rt = w.return_time();
        assert!(rt > 0 || w.position == 0);
    }

    #[test]
    fn test_ensemble_occupation() {
        let mut ens = WalkEnsemble::new(50, 0);
        let rng_vals: Vec<f64> = (0..50000).map(|i| ((i * 31 + 17) as f64 * 0.00001) % 1.0).collect();
        let mut idx = 0;
        for _ in 0..100 {
            for w in &mut ens.walks {
                w.step_simple(rng_vals[idx % rng_vals.len()]);
                idx += 1;
            }
        }
        let occ = ens.mean_occupation();
        for state in [-1i8, 0, 1] {
            let f = occ.get(&state).copied().unwrap_or(0.0);
            assert!(f > 0.25, "State {} should be > 0.25, got {}", state, f);
        }
    }

    #[test]
    fn test_autocorrelation_simple_walk_is_small() {
        // For a simple symmetric walk on Z3, autocorrelation should be near zero
        let mut w = TernaryWalk::new(0);
        for i in 0..5000 {
            w.step_simple(((i * 23 + 7) as f64 * 0.0001) % 1.0);
        }
        let ens = WalkEnsemble { walks: vec![TernaryWalk { position: w.position, history: w.history.clone(), step_count: w.step_count }] };
        let ac1 = ens.autocorrelation(1);
        // Simple symmetric walk on Z3: autocorrelation near zero
        assert!(ac1.abs() < 0.3, "Simple walk autocorrelation should be small: {}", ac1);
    }
}
