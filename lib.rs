#![forbid(unsafe_code)]

/// Random walks on ternary lattices.

#[derive(Debug, Clone)]
pub struct Walk {
    pub position: u32,
    pub state: i8,
    pub steps: u32,
}

impl Walk {
    pub fn new(start: usize, state: i8) -> Self {
        Walk { position: start as u32, state, steps: 0 }
    }

    /// Move one step in a cardinal direction. direction: 0=up, 1=right, 2=down, 3=left.
    pub fn step(&mut self, width: usize, direction: u8) {
        let w = width as u32;
        match direction % 4 {
            0 => { if self.position >= w { self.position -= w; } }
            1 => { self.position += 1; }
            2 => { self.position += w; }
            3 => { if self.position > 0 { self.position -= 1; } }
            _ => {}
        }
        self.steps += 1;
    }
}

/// Generate a random walk path of `steps` steps using the provided closure as RNG.
/// The closure should return a u8 in 0..4.
pub fn random_walk<F: FnMut() -> u8>(start: usize, steps: u64, width: usize, mut rng: F) -> Vec<usize> {
    let mut walk = Walk::new(start, 0);
    let mut path = vec![start];
    for _ in 0..steps {
        walk.step(width, rng());
        path.push(walk.position as usize);
    }
    path
}

/// Run walks from multiple starting positions and return histogram of final positions.
pub fn walk_distribution<F: FnMut() -> u8>(starters: &[usize], steps: u64, width: usize, mut rng: F) -> Vec<usize> {
    let mut hist: Vec<usize> = vec![0; starters.iter().map(|&s| s + steps as usize * width + steps as usize + 10).max().unwrap_or(100)];
    for &start in starters {
        let path = random_walk(start, steps, width, &mut rng);
        let final_pos = *path.last().unwrap();
        if final_pos < hist.len() {
            hist[final_pos] += 1;
        }
    }
    // Trim trailing zeros
    while hist.len() > 1 && *hist.last().unwrap() == 0 {
        hist.pop();
    }
    hist
}

/// Compute mean squared displacement from paths.
pub fn mean_squared_displacement(paths: &[Vec<usize>]) -> f64 {
    if paths.is_empty() { return 0.0; }
    let total: f64 = paths.iter().map(|path| {
        if path.len() < 2 { return 0.0f64; }
        let start = path[0] as f64;
        let end = *path.last().unwrap() as f64;
        (end - start).powi(2)
    }).sum();
    total / paths.len() as f64
}

/// Compute anomalous diffusion exponent from paths.
/// Fits MSD ∝ t^α. Returns α.
fn _simple_paths_msd_at_step(paths: &[Vec<usize>], t: usize) -> f64 {
    if paths.is_empty() || t == 0 { return 0.0; }
    let total: f64 = paths.iter().filter(|p| p.len() > t).map(|path| {
        let start = path[0] as f64;
        let pos = path[t] as f64;
        (pos - start).powi(2)
    }).sum();
    let count = paths.iter().filter(|p| p.len() > t).count() as f64;
    if count == 0.0 { 0.0 } else { total / count }
}

/// Compute anomalous exponent. Returns α where MSD ~ t^α.
/// 1.0 = normal diffusion, >1 superdiffusive, <1 subdiffusive.
pub fn anomalous_exponent(paths: &[Vec<usize>]) -> f64 {
    if paths.is_empty() { return 0.0; }
    let min_len = paths.iter().map(|p| p.len()).min().unwrap_or(0);
    if min_len < 3 { return 0.0; }

    // Sample MSD at a few time points and fit log-log slope
    let t1 = 1;
    let t2 = (min_len - 1) / 2;
    let t3 = min_len - 1;
    if t2 <= t1 || t3 <= t2 { return 0.0; }

    let msd1 = _simple_paths_msd_at_step(paths, t1);
    let msd2 = _simple_paths_msd_at_step(paths, t2);
    let msd3 = _simple_paths_msd_at_step(paths, t3);

    if msd1 <= 0.0 || msd2 <= 0.0 || msd3 <= 0.0 { return 0.0; }

    // Linear regression in log-log space
    let log_t = [t1 as f64, t2 as f64, t3 as f64].map(|x| x.ln());
    let log_msd = [msd1, msd2, msd3].map(|x| x.ln());
    let n = 3.0f64;
    let sum_x: f64 = log_t.iter().sum();
    let sum_y: f64 = log_msd.iter().sum();
    let sum_xy: f64 = log_t.iter().zip(log_msd.iter()).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = log_t.iter().map(|x| x * x).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom == 0.0 { return 0.0; }
    (n * sum_xy - sum_x * sum_y) / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_new() {
        let w = Walk::new(5, 1);
        assert_eq!(w.position, 5);
        assert_eq!(w.state, 1);
        assert_eq!(w.steps, 0);
    }

    #[test]
    fn test_walk_step_right() {
        let mut w = Walk::new(5, 0);
        w.step(10, 1); // right
        assert_eq!(w.position, 6);
        assert_eq!(w.steps, 1);
    }

    #[test]
    fn test_walk_step_left() {
        let mut w = Walk::new(5, 0);
        w.step(10, 3); // left
        assert_eq!(w.position, 4);
    }

    #[test]
    fn test_walk_step_up() {
        let mut w = Walk::new(15, 0);
        w.step(10, 0); // up
        assert_eq!(w.position, 5);
    }

    #[test]
    fn test_walk_step_down() {
        let mut w = Walk::new(5, 0);
        w.step(10, 2); // down
        assert_eq!(w.position, 15);
    }

    #[test]
    fn test_walk_struct_size() {
        assert!(std::mem::size_of::<Walk>() <= 16);
    }

    #[test]
    fn test_random_walk_length() {
        let mut counter = 0u8;
        let path = random_walk(5, 10, 100, || { counter += 1; counter % 4 });
        assert_eq!(path.len(), 11); // start + 10 steps
    }

    #[test]
    fn test_random_walk_starts_at_start() {
        let path = random_walk(42, 5, 100, || 0);
        assert_eq!(path[0], 42);
    }

    #[test]
    fn test_random_walk_all_up() {
        let path = random_walk(50, 5, 10, || 0); // all up
        assert_eq!(path[5], 0); // 50 - 5*10 = 0
    }

    #[test]
    fn test_random_walk_all_right() {
        let path = random_walk(5, 3, 100, || 1); // all right
        assert_eq!(path[3], 8); // 5 + 3
    }

    #[test]
    fn test_mean_squared_displacement_zero() {
        let paths = vec![vec![5, 5, 5, 5]]; // no movement
        assert_eq!(mean_squared_displacement(&paths), 0.0);
    }

    #[test]
    fn test_mean_squared_displacement_linear() {
        let paths = vec![vec![0, 1, 2, 3, 4]]; // all right, width=100
        let msd = mean_squared_displacement(&paths);
        assert_eq!(msd, 16.0); // (4-0)^2
    }

    #[test]
    fn test_mean_squared_displacement_multiple() {
        let paths = vec![vec![0, 4], vec![0, 8]];
        let msd = mean_squared_displacement(&paths);
        assert_eq!(msd, 40.0); // (16 + 64) / 2
    }

    #[test]
    fn test_walk_distribution() {
        let mut vals = [0u8, 1, 2, 3];
        let mut idx = 0;
        let dist = walk_distribution(&[50], 4, 10, || {
            let v = vals[idx % 4];
            idx += 1;
            v
        });
        // Walk from 50: up→40, right→41, down→51, left→50 → final=50
        assert!(dist.len() > 50);
        assert!(dist[50] >= 1);
    }
}
