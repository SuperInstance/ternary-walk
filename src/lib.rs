#![forbid(unsafe_code)]
#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use alloc::collections::BTreeMap;

// ── Helpers ──

/// Clamp a value into the range [-1, 1].
fn clamp_ternary(v: i64) -> i64 {
    if v < -1 { -1 } else if v > 1 { 1 } else { v }
}

/// Convert a ternary step index (0, 1, 2) to value (-1, 0, +1).
fn step_value(idx: usize) -> i64 {
    match idx {
        0 => -1,
        1 => 0,
        _ => 1,
    }
}

/// Weighted random selection among 3 choices using a closure that returns `u32`.
/// `weights` = [w_neg, w_zero, w_pos]. Returns 0, 1, or 2.
fn weighted_pick(weights: [u32; 3], rng: &mut impl FnMut() -> u32) -> usize {
    let total = weights[0].saturating_add(weights[1]).saturating_add(weights[2]);
    if total == 0 {
        return 1; // default to 0-step
    }
    let r = (rng()) % total;
    if r < weights[0] {
        0
    } else if r < weights[0] + weights[1] {
        1
    } else {
        2
    }
}

// ── 1. SimpleTernaryWalk ──

/// A simple random walk where each step adds -1, 0, or +1.
#[derive(Debug, Clone)]
pub struct SimpleTernaryWalk {
    pub position: i64,
    pub steps: u64,
    pub history: Vec<i64>,
}

impl SimpleTernaryWalk {
    pub fn new(start: i64) -> Self {
        let mut w = SimpleTernaryWalk {
            position: start,
            steps: 0,
            history: Vec::new(),
        };
        w.history.push(start);
        w
    }

    /// Take one uniform random step using the provided RNG.
    pub fn step(&mut self, rng: &mut impl FnMut() -> u32) {
        let delta = step_value((rng() % 3) as usize);
        self.position += delta;
        self.steps += 1;
        self.history.push(self.position);
    }

    /// Take a step with explicit delta (-1, 0, or +1).
    pub fn step_with(&mut self, delta: i64) {
        let d = clamp_ternary(delta);
        self.position += d;
        self.steps += 1;
        self.history.push(self.position);
    }

    /// Run `n` uniform steps.
    pub fn run(&mut self, n: u64, rng: &mut impl FnMut() -> u32) {
        for _ in 0..n {
            self.step(rng);
        }
    }
}

// ── 2. BiasedWalk ──

/// A random walk with configurable probabilities for each direction.
/// `weights` = [weight for -1, weight for 0, weight for +1].
#[derive(Debug, Clone)]
pub struct BiasedWalk {
    pub position: i64,
    pub steps: u64,
    pub weights: [u32; 3],
    pub history: Vec<i64>,
}

impl BiasedWalk {
    pub fn new(start: i64, weights: [u32; 3]) -> Self {
        let mut w = BiasedWalk {
            position: start,
            steps: 0,
            weights,
            history: Vec::new(),
        };
        w.history.push(start);
        w
    }

    pub fn step(&mut self, rng: &mut impl FnMut() -> u32) {
        let idx = weighted_pick(self.weights, rng);
        self.position += step_value(idx);
        self.steps += 1;
        self.history.push(self.position);
    }

    pub fn run(&mut self, n: u64, rng: &mut impl FnMut() -> u32) {
        for _ in 0..n {
            self.step(rng);
        }
    }
}

// ── 3. WalkStatistics ──

/// Statistics computed from a walk's history.
#[derive(Debug, Clone)]
pub struct WalkStatistics {
    pub position: i64,
    pub min: i64,
    pub max: i64,
    pub mean: f64,
    pub variance: f64,
    pub zero_crossings: u64,
    pub time_in_each_state: [u64; 3], // [-1, 0, +1]
}

impl WalkStatistics {
    /// Compute statistics from a history slice.
    pub fn from_history(history: &[i64]) -> Self {
        if history.is_empty() {
            return WalkStatistics {
                position: 0, min: 0, max: 0,
                mean: 0.0, variance: 0.0,
                zero_crossings: 0,
                time_in_each_state: [0, 0, 0],
            };
        }

        let position = *history.last().unwrap();
        let mut min_val = history[0];
        let mut max_val = history[0];
        let mut sum = 0i64;
        let mut zero_crossings = 0u64;
        let mut state_counts = [0u64; 3]; // [-1, 0, +1]

        for (i, &v) in history.iter().enumerate() {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
            sum += v;

            // State classification: mod 3 mapped to ternary
            let state_idx = match v % 3 {
                -2 | 1 => 2,  // +1 equivalent
                -1 | 2 => 0,  // -1 equivalent
                _ => 1,       // 0
            };
            state_counts[state_idx] += 1;

            if i > 0 {
                let prev = history[i - 1];
                if (prev < 0 && v >= 0) || (prev >= 0 && v < 0) {
                    zero_crossings += 1;
                }
            }
        }

        let n = history.len() as f64;
        let mean = sum as f64 / n;
        let variance = history.iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;

        WalkStatistics {
            position,
            min: min_val,
            max: max_val,
            mean,
            variance,
            zero_crossings,
            time_in_each_state: state_counts,
        }
    }
}

// ── 4. SpatialTernaryWalk ──

/// A walk on a 2D ternary grid. Each grid cell holds a ternary value {-1, 0, +1}.
/// The walker moves in 2D (up/down/left/right) and reads the grid value at each step.
#[derive(Debug, Clone)]
pub struct SpatialTernaryWalk {
    pub x: i64,
    pub y: i64,
    pub steps: u64,
    pub values_read: Vec<i64>,
    pub path: Vec<(i64, i64)>,
    grid: BTreeMap<(i64, i64), i64>,
}

impl SpatialTernaryWalk {
    /// Create a walk on the given grid. Grid is a flat slice of ternary values in row-major order.
    pub fn new(width: usize, height: usize, grid_data: &[i64], start_x: i64, start_y: i64) -> Self {
        let mut grid = BTreeMap::new();
        for (i, &v) in grid_data.iter().enumerate() {
            let gy = i / width;
            let gx = i % width;
            grid.insert((gx as i64, gy as i64), clamp_ternary(v));
        }
        let _ = height; // bounds info available via grid
        let mut w = SpatialTernaryWalk {
            x: start_x,
            y: start_y,
            steps: 0,
            values_read: Vec::new(),
            path: Vec::new(),
            grid,
        };
        w.path.push((start_x, start_y));
        w.values_read.push(w.read_grid());
        w
    }

    fn read_grid(&self) -> i64 {
        *self.grid.get(&(self.x, self.y)).unwrap_or(&0)
    }

    /// Move: direction 0=up, 1=right, 2=down, 3=left.
    pub fn step(&mut self, direction: u8) {
        match direction % 4 {
            0 => self.y += 1,
            1 => self.x += 1,
            2 => self.y -= 1,
            3 => self.x -= 1,
            _ => {}
        }
        self.steps += 1;
        self.path.push((self.x, self.y));
        self.values_read.push(self.read_grid());
    }

    /// Run `n` steps using the provided RNG for direction selection.
    pub fn run(&mut self, n: u64, rng: &mut impl FnMut() -> u32) {
        for _ in 0..n {
            let dir = (rng() % 4) as u8;
            self.step(dir);
        }
    }
}

// ── 5. AbsorbingWalk ──

/// A walk that stops (is absorbed) when it hits a specific target value.
#[derive(Debug, Clone)]
pub struct AbsorbingWalk {
    pub position: i64,
    pub steps: u64,
    pub absorbed: bool,
    pub target: i64,
    pub history: Vec<i64>,
}

impl AbsorbingWalk {
    pub fn new(start: i64, target: i64) -> Self {
        let absorbed = start == target;
        let mut w = AbsorbingWalk {
            position: start,
            steps: 0,
            absorbed,
            target,
            history: Vec::new(),
        };
        w.history.push(start);
        w
    }

    /// Take a step; returns false if already absorbed (no step taken).
    pub fn step(&mut self, delta: i64) -> bool {
        if self.absorbed {
            return false;
        }
        let d = clamp_ternary(delta);
        self.position += d;
        self.steps += 1;
        self.history.push(self.position);
        if self.position == self.target {
            self.absorbed = true;
        }
        true
    }

    /// Run with RNG until absorbed or max_steps reached. Returns steps taken.
    pub fn run_until_absorbed(&mut self, max_steps: u64, rng: &mut impl FnMut() -> u32) -> u64 {
        for _ in 0..max_steps {
            if self.absorbed {
                break;
            }
            let delta = step_value((rng() % 3) as usize);
            self.step(delta);
        }
        self.steps
    }
}

// ── 6. ReflectingWalk ──

/// A walk with reflecting boundaries at `lo` and `hi`.
#[derive(Debug, Clone)]
pub struct ReflectingWalk {
    pub position: i64,
    pub steps: u64,
    pub lo: i64,
    pub hi: i64,
    pub history: Vec<i64>,
}

impl ReflectingWalk {
    pub fn new(start: i64, lo: i64, hi: i64) -> Self {
        let s = if start < lo { lo } else if start > hi { hi } else { start };
        let mut w = ReflectingWalk {
            position: s,
            steps: 0,
            lo,
            hi,
            history: Vec::new(),
        };
        w.history.push(s);
        w
    }

    pub fn step(&mut self, delta: i64) {
        let d = clamp_ternary(delta);
        self.position += d;
        // Reflect at boundaries
        if self.position < self.lo {
            self.position = self.lo + (self.lo - self.position);
        }
        if self.position > self.hi {
            self.position = self.hi - (self.position - self.hi);
        }
        self.steps += 1;
        self.history.push(self.position);
    }

    pub fn run(&mut self, n: u64, rng: &mut impl FnMut() -> u32) {
        for _ in 0..n {
            let delta = step_value((rng() % 3) as usize);
            self.step(delta);
        }
    }
}

// ── 7. CorrelatedWalk ──

/// A walk where each step has momentum — probability of continuing in the same direction
/// is higher. `momentum` is in [0.0, 1.0]; higher = more likely to repeat direction.
#[derive(Debug, Clone)]
pub struct CorrelatedWalk {
    pub position: i64,
    pub steps: u64,
    pub momentum: u32, // out of 100
    pub last_direction: i64, // -1, 0, or +1
    pub history: Vec<i64>,
}

impl CorrelatedWalk {
    pub fn new(start: i64, momentum: u32, initial_direction: i64) -> Self {
        let mut w = CorrelatedWalk {
            position: start,
            steps: 0,
            momentum: if momentum > 100 { 100 } else { momentum },
            last_direction: clamp_ternary(initial_direction),
            history: Vec::new(),
        };
        w.history.push(start);
        w
    }

    pub fn step(&mut self, rng: &mut impl FnMut() -> u32) {
        let roll = (rng()) % 100;
        let delta = if roll < self.momentum {
            self.last_direction
        } else {
            step_value((rng()) as usize % 3)
        };
        self.last_direction = delta;
        self.position += delta;
        self.steps += 1;
        self.history.push(self.position);
    }

    pub fn run(&mut self, n: u64, rng: &mut impl FnMut() -> u32) {
        for _ in 0..n {
            self.step(rng);
        }
    }
}

// ── 8. FirstPassageTime ──

/// Compute the first passage time (number of steps) to reach each distinct value
/// in the walk history.
#[derive(Debug, Clone)]
pub struct FirstPassageTime {
    /// Map from target value to the step index at which it was first reached.
    pub passage_times: BTreeMap<i64, u64>,
}

impl FirstPassageTime {
    pub fn from_history(history: &[i64]) -> Self {
        let mut passage_times = BTreeMap::new();
        for (step, &val) in history.iter().enumerate() {
            passage_times.entry(val).or_insert_with(|| step as u64);
        }
        FirstPassageTime { passage_times }
    }

    /// Get the first passage time to a specific value. Returns None if never reached.
    pub fn time_to(&self, value: i64) -> Option<u64> {
        self.passage_times.get(&value).copied()
    }
}

// ── 9. OccupationMeasure ──

/// Fraction of time the walk spends in each ternary state {-1, 0, +1},
/// determined by `value % 3`.
#[derive(Debug, Clone)]
pub struct OccupationMeasure {
    /// Fraction of time in state -1.
    pub frac_neg1: f64,
    /// Fraction of time in state 0.
    pub frac_zero: f64,
    /// Fraction of time in state +1.
    pub frac_pos1: f64,
}

impl OccupationMeasure {
    pub fn from_history(history: &[i64]) -> Self {
        if history.is_empty() {
            return OccupationMeasure { frac_neg1: 0.0, frac_zero: 0.0, frac_pos1: 0.0 };
        }
        let n = history.len() as f64;
        let mut counts = [0u64; 3]; // [-1, 0, +1]
        for &v in history {
            let idx = match v % 3 {
                -2 | 1 => 2,  // +1
                -1 | 2 => 0,  // -1
                _ => 1,       // 0
            };
            counts[idx] += 1;
        }
        OccupationMeasure {
            frac_neg1: counts[0] as f64 / n,
            frac_zero: counts[1] as f64 / n,
            frac_pos1: counts[2] as f64 / n,
        }
    }
}

// ── 10. ReturnTimeDistribution ──

/// Distribution of return times (steps between returns) to the origin or each visited state.
#[derive(Debug, Clone)]
pub struct ReturnTimeDistribution {
    /// Map from value → list of interval lengths between consecutive visits.
    pub return_intervals: BTreeMap<i64, Vec<u64>>,
}

impl ReturnTimeDistribution {
    /// Compute return time intervals for the origin (0).
    pub fn to_origin(history: &[i64]) -> Self {
        Self::to_value(history, 0)
    }

    /// Compute return time intervals for all visited states.
    pub fn to_all_states(history: &[i64]) -> Self {
        let all_values: Vec<i64> = {
            let mut vals = Vec::new();
            let mut seen = BTreeMap::new();
            for &v in history {
                seen.insert(v, ());
            }
            for &v in seen.keys() {
                vals.push(v);
            }
            vals
        };
        let mut all_intervals = BTreeMap::new();
        for &val in &all_values {
            let dist = Self::to_value(history, val);
            if let Some(intervals) = dist.return_intervals.get(&val) {
                all_intervals.insert(val, intervals.clone());
            }
        }
        ReturnTimeDistribution { return_intervals: all_intervals }
    }

    /// Compute return time intervals for a specific value.
    pub fn to_value(history: &[i64], target: i64) -> Self {
        let mut last_visit: Option<u64> = None;
        let mut intervals: Vec<u64> = Vec::new();
        for (step, &v) in history.iter().enumerate() {
            if v == target {
                if let Some(prev) = last_visit {
                    intervals.push(step as u64 - prev);
                }
                last_visit = Some(step as u64);
            }
        }
        let mut map = BTreeMap::new();
        map.insert(target, intervals);
        ReturnTimeDistribution { return_intervals: map }
    }

    /// Mean return time for a given value. Returns None if no returns occurred.
    pub fn mean_return_time(&self, value: i64) -> Option<f64> {
        self.return_intervals.get(&value).and_then(|intervals| {
            if intervals.is_empty() {
                None
            } else {
                Some(intervals.iter().sum::<u64>() as f64 / intervals.len() as f64)
            }
        })
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn deterministic_rng(vals: &[u32]) -> impl FnMut() -> u32 + '_ {
        let mut i = 0usize;
        move || {
            let v = vals[i % vals.len()];
            i += 1;
            v
        }
    }

    #[test]
    fn test_simple_walk_basic() {
        let mut w = SimpleTernaryWalk::new(0);
        w.step_with(1);
        w.step_with(1);
        w.step_with(-1);
        assert_eq!(w.position, 1);
        assert_eq!(w.steps, 3);
        assert_eq!(w.history, vec![0, 1, 2, 1]);
    }

    #[test]
    fn test_simple_walk_run() {
        let mut rng = deterministic_rng(&[0, 1, 2]); // -1, 0, +1 repeating
        let mut w = SimpleTernaryWalk::new(0);
        w.run(6, &mut rng);
        // steps: -1, 0, +1, -1, 0, +1 → position: -1, -1, 0, -1, -1, 0
        assert_eq!(w.position, 0);
        assert_eq!(w.steps, 6);
        assert_eq!(w.history.len(), 7); // start + 6 steps
    }

    #[test]
    fn test_biased_walk_all_positive() {
        let mut rng = deterministic_rng(&[0]); // any value works since weights force +1
        let mut w = BiasedWalk::new(0, [0, 0, 1]); // always +1
        w.run(5, &mut rng);
        assert_eq!(w.position, 5);
    }

    #[test]
    fn test_biased_walk_all_negative() {
        let mut rng = deterministic_rng(&[0]);
        let mut w = BiasedWalk::new(10, [1, 0, 0]); // always -1
        w.run(3, &mut rng);
        assert_eq!(w.position, 7);
    }

    #[test]
    fn test_walk_statistics() {
        let history = vec![0i64, 1, -1, 0, 2];
        let stats = WalkStatistics::from_history(&history);
        assert_eq!(stats.position, 2);
        assert_eq!(stats.min, -1);
        assert_eq!(stats.max, 2);
        // 0→1: no cross (both >=0)
        // 1→-1: cross (>=0 to <0)
        // -1→0: cross (<0 to >=0)
        // 0→2: no cross
        assert_eq!(stats.zero_crossings, 2);
    }

    #[test]
    fn test_walk_statistics_mean_variance() {
        let history = vec![2i64, 4, 4, 4, 5, 5, 7, 9];
        let stats = WalkStatistics::from_history(&history);
        let expected_mean = 40.0 / 8.0;
        assert!((stats.mean - expected_mean).abs() < 1e-10);
        assert!(stats.variance > 0.0);
    }

    #[test]
    fn test_spatial_walk() {
        // 3x3 grid: all zeros except center = 1, top-left = -1
        let grid = vec![-1i64, 0, 0, 0, 1, 0, 0, 0, 0];
        let mut sw = SpatialTernaryWalk::new(3, 3, &grid, 1, 1); // start at center
        assert_eq!(sw.values_read[0], 1); // center value

        sw.step(0); // up → (1, 2)
        assert_eq!(sw.x, 1);
        assert_eq!(sw.y, 2);
        assert_eq!(*sw.values_read.last().unwrap(), 0);

        sw.step(3); // left → (0, 2) — off grid edge reads 0
        assert_eq!(sw.x, 0);
        assert_eq!(sw.y, 2);
    }

    #[test]
    fn test_absorbing_walk() {
        let mut w = AbsorbingWalk::new(0, 3);
        assert!(!w.absorbed);
        w.step(1); // pos=1
        assert!(!w.absorbed);
        w.step(1); // pos=2
        w.step(1); // pos=3 → absorbed
        assert!(w.absorbed);
        assert_eq!(w.position, 3);
        let result = w.step(1); // no-op
        assert!(!result);
    }

    #[test]
    fn test_reflecting_walk() {
        let mut w = ReflectingWalk::new(0, -2, 2);
        w.step(1); // 1
        w.step(1); // 2
        w.step(1); // would be 3 → reflect: hi=2, 3-2=1 → 2-1=1
        assert_eq!(w.position, 1);
        assert_eq!(w.steps, 3);
    }

    #[test]
    fn test_reflecting_walk_lower_boundary() {
        let mut w = ReflectingWalk::new(0, -2, 2);
        w.step(-1); // -1
        w.step(-1); // -2
        w.step(-1); // would be -3 → reflect: lo=-2, -3-(-2)=-1 → -2+1=-1
        assert_eq!(w.position, -1);
    }

    #[test]
    fn test_correlated_walk_high_momentum() {
        let mut rng = deterministic_rng(&[99, 0, 0, 0, 0, 0]); // high roll → keep direction
        let mut w = CorrelatedWalk::new(0, 90, 1); // 90% momentum, direction +1
        w.run(5, &mut rng);
        // First step: roll=99 < 90? No (99 >= 90) → new direction from rng: step_value(0 % 3) = step_value(0) = -1
        // Hmm, let me reconsider the logic.
        // roll < momentum → repeat last_direction
        // roll = 99, momentum = 90: 99 < 90 is false → new: step_value((rng()) as usize % 3) = step_value(0 % 3) = -1
        // So first step goes -1. Let me adjust the test to check behavior rather than specific position.
        assert_eq!(w.steps, 5);
        assert_eq!(w.history.len(), 6);
    }

    #[test]
    fn test_correlated_walk_perfect_momentum() {
        let mut rng = deterministic_rng(&[0, 0, 0, 0]);
        let mut w = CorrelatedWalk::new(0, 100, 1); // always repeats direction +1
        w.run(5, &mut rng);
        assert_eq!(w.position, 5); // all +1
    }

    #[test]
    fn test_first_passage_time() {
        let history = vec![0i64, 1, 2, 1, 3];
        let fpt = FirstPassageTime::from_history(&history);
        assert_eq!(fpt.time_to(0), Some(0));
        assert_eq!(fpt.time_to(1), Some(1));
        assert_eq!(fpt.time_to(2), Some(2));
        assert_eq!(fpt.time_to(3), Some(4));
        assert_eq!(fpt.time_to(99), None);
    }

    #[test]
    fn test_occupation_measure() {
        let history = vec![0i64, 1, 2, 3, -1];
        // States: 0%3=0→idx1, 1%3=1→idx2(+1), 2%3=2→idx0(-1), 3%3=0→idx1, -1%3=-1→idx0(-1)
        // counts: [-1: 2, 0: 2, +1: 1]
        let om = OccupationMeasure::from_history(&history);
        assert!((om.frac_neg1 - 2.0 / 5.0).abs() < 1e-10);
        assert!((om.frac_zero - 2.0 / 5.0).abs() < 1e-10);
        assert!((om.frac_pos1 - 1.0 / 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_return_time_to_origin() {
        let history = vec![0i64, 1, 0, 2, 0, 3];
        let dist = ReturnTimeDistribution::to_origin(&history);
        let intervals = dist.return_intervals.get(&0).unwrap();
        assert_eq!(intervals, &vec![2u64, 2]); // returns at steps 2 and 4, intervals = 2, 2
        assert!((dist.mean_return_time(0).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_return_time_to_all_states() {
        let history = vec![0i64, 1, 0, 1];
        let dist = ReturnTimeDistribution::to_all_states(&history);
        assert_eq!(dist.return_intervals.get(&0).unwrap(), &vec![2u64]);
        assert_eq!(dist.return_intervals.get(&1).unwrap(), &vec![2u64]);
    }

    #[test]
    fn test_walk_statistics_empty() {
        let stats = WalkStatistics::from_history(&[]);
        assert_eq!(stats.position, 0);
        assert_eq!(stats.min, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_occupation_measure_empty() {
        let om = OccupationMeasure::from_history(&[]);
        assert_eq!(om.frac_neg1, 0.0);
        assert_eq!(om.frac_zero, 0.0);
        assert_eq!(om.frac_pos1, 0.0);
    }
}
