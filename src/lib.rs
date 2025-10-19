use libm::erfc;
use nanorand::*;

#[cfg(feature = "num-complex")]
use num_complex::*;

// Ziggurat algorithm
// use C = 256 blocks
// Each section has area V, 1.0 / 256
// Each rectangle has area V except bottom R0, which is V - tail

#[derive(Clone, Debug)]
pub struct Ziggurat {
    c: u32,
    r: f64,
    v: f64,
    xs: Vec<f64>,
    f: fn(f64) -> f64,
}

impl Ziggurat {
    pub fn new(c: u32) -> Self {
        // Our one sided pdf, f(x), defined on [0..infinity)
        let f = |x: f64| (-0.5 * x.powi(2)).exp();
        // integral of f(u) from u=x..infinity
        let integral = |x: f64| std::f64::consts::FRAC_PI_2.sqrt() * erfc(x / 2_f64.sqrt());

        let (_xc, r) = Self::find_r(c);
        let v = r * f(r) + integral(r);
        let xs = Self::find_xs(c);
        Self { c, r, v, xs, f }
    }
    pub fn find_r(c: u32) -> (f64, f64) {
        let f = |x: f64| (-0.5 * x.powi(2)).exp();

        // The inverse pdf, f^-1(x), defined on (0, 1].
        let finv = |x: f64| (-2.0 * x.ln()).sqrt();

        // integral of f(u) from u=x..infinity
        let _numeric_integral = |x: f64| {
            let step = 1e-4;
            let mut output = 0.0;
            let mut left = x;
            loop {
                let new = step * 0.5 * (f(left) + f(left + step));
                left += step;
                output += new;
                if new < 1e-13 {
                    break;
                }
            }
            output
        };

        // integral of f(u) from u=x..infinity
        let integral = |x: f64| std::f64::consts::FRAC_PI_2.sqrt() * erfc(x / 2_f64.sqrt());

        let lower_bound = integral(0.0);
        let calculate_xc = |r: f64| {
            // Given a candidate r, we just solve for V(r)
            let vr = r * f(r) + integral(r);
            if c as f64 * vr < lower_bound {
                // C * V(r) should be lower bounded by integral of f(x), x=0..infinity since we're summing area of
                // rectangles that cover this region.
                return None;
            }
            let xnext = |x: f64| {
                let tmp = f(x) + vr / x;
                if tmp <= 0.0 || tmp > 1.0 {
                    None
                } else {
                    Some(finv(tmp))
                }
            };
            let mut xi = r;
            for _i in 2..c + 1 {
                // i represents which x index, so we initialize xi = r for i = 1, and then on the first iteration of this
                // loop we are calculating xi for i = 2. We want xi = 0.0 for i = C.
                let tmp = xnext(xi);
                if let Some(value) = tmp {
                    if value > xi {
                        panic!("xi should be monotonic decreasing");
                    }
                    xi = value;
                } else {
                    // Tried to hand an invalid input value to invf(x), this candidate r is bad.
                    return None;
                }
            }
            Some(xi)
        };

        // Solve for r by strobing uniformly
        let mut alpha = 1e-1;
        let mut r = 0.0;
        let mut best_xc = 1000.0;
        let mut best_r = r;

        // Calculate an initial point
        let mut r0 = r;
        loop {
            r += 1e-4;
            let vr = r * f(r) + integral(r);
            if c as f64 * vr < lower_bound {
                // C * V(r) should be lower bounded by integral of f(x), x=0..infinity since we're summing area of
                // rectangles that cover this region.
                break;
            }
            r0 = r;
        }

        let mut xc0 = calculate_xc(r0).unwrap();

        r -= 2e-4;
        let mut drdxc = 0.0;
        let mut epsilon_mode_engage = false;
        loop {
            // Check stopping criteria
            let vr = r * f(r) + integral(r);
            if c as f64 * vr < lower_bound {
                // C * V(r) should be lower bounded by integral of f(x), x=0..infinity since we're summing area of
                // rectangles that cover this region.
                panic!("should be close enough at this point that we don't hit this error");
            }
            if let Some(xc) = calculate_xc(r) {
                // Now that we have xi = x[c], see if it's closer to goal (0.0) than prior best.
                if xc < best_xc {
                    best_xc = xc;
                    best_r = r;
                }

                // Update our gradient parameters
                let dr = r - r0;
                let dxc = xc - xc0;
                r0 = r;
                xc0 = xc;
                drdxc = dr / dxc;
                if dr.abs() < 1e-14 {
                    epsilon_mode_engage = true;
                }
                if !epsilon_mode_engage {
                    r -= alpha * drdxc;
                } else {
                    r -= r * f64::EPSILON;
                }
            } else {
                if epsilon_mode_engage {
                    // Ok there's nothing else we can do, just end it.
                    break;
                }
                r += alpha * drdxc;
                alpha *= 0.5;
                r -= alpha * drdxc;
            }
        }

        (best_xc, best_r)
    }

    pub fn find_xs(c: u32) -> Vec<f64> {
        // Our one sided pdf, f(x), defined on [0..infinity)
        let f = |x: f64| (-0.5 * x.powi(2)).exp();

        // The inverse pdf, f^-1(x), defined on (0, 1].
        let finv = |x: f64| (-2.0 * x.ln()).sqrt();

        // integral of f(u) from u=x..infinity
        let integral = |x: f64| std::f64::consts::FRAC_PI_2.sqrt() * erfc(x / 2_f64.sqrt());

        let (_xc, r) = Self::find_r(c);
        let vr = r * f(r) + integral(r);
        let xnext = |x: f64| {
            let tmp = f(x) + vr / x;
            if tmp <= 0.0 || tmp > 1.0 {
                None
            } else {
                Some(finv(tmp))
            }
        };

        // Collect the x's.
        let mut output = vec![r];
        let mut xi = r;
        for _i in 2..c {
            xi = xnext(xi).unwrap();
            output.push(xi);
        }
        output.push(0.0); // x[c] = 0.0 by definition.
        output
    }
}

pub struct Awgn {
    rng: WyRand,
    ziggurat: Ziggurat,
}

impl Awgn {
    pub fn new(seed: u64, c: u32) -> Self {
        Self {
            rng: WyRand::new_seed(seed),
            ziggurat: Ziggurat::new(c),
        }
    }

    #[inline]
    pub fn rand_f64(&mut self) -> f64 {
        f64::from_bits((self.rng.generate::<u64>() & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000)
            - 1.0
    }

    #[inline]
    pub fn rand_f32(&mut self) -> f32 {
        f32::from_bits(self.rng.generate::<u32>() & 0x007f_ffff | 0x3f80_0000) - 1.0
    }

    pub fn rand_vec_f64(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.rand_f64()).collect()
    }

    pub fn rand_vec_f32(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.rand_f32()).collect()
    }

    pub fn fill_f32(&mut self, buffer: &mut [f32]) {
        for x in buffer {
            *x = self.rand_f32();
        }
    }

    pub fn fill_f64(&mut self, buffer: &mut [f64]) {
        for x in buffer {
            *x = self.rand_f64();
        }
    }

    /* Simple Box-Muller Transform
    z1 = sqrt { -2 ln x1 } * cos { 2 * pi * x2 }
    z2 = sqrt { -2 ln x1 } * sin { 2 * pi * x2 }
    */
    #[cfg(feature = "num-complex")]
    pub fn crandn_box_muller_f32(&mut self) -> Complex<f32> {
        let mag = (-2.0 * self.rand_f32().ln()).sqrt();
        let rad = 2.0 * std::f32::consts::PI * self.rand_f32();
        Complex::new(mag * (rad.cos()), mag * (rad.sin()))
    }

    #[cfg(feature = "num-complex")]
    pub fn crandn_box_muller_f64(&mut self) -> Complex<f64> {
        let mag = (-2.0 * self.rand_f64().ln()).sqrt();
        let rad = 2.0 * std::f64::consts::PI * self.rand_f64();
        Complex::new(mag * (rad.cos()), mag * (rad.sin()))
    }

    pub fn randn_f64(&mut self) -> f64 {
        let c = self.ziggurat.c;

        // Draw samples until one is accepted. Should be the first one most of the time.
        loop {
            // Select a random box with probability 1/c, Bi
            let idx = self.rng.generate_range(0_usize..c as usize);

            // Draw a random number z in the box as
            let u0 = 2.0 * self.rand_f64() - 1.0;
            let z = if idx == 0 {
                // z = U0 * V / f(x1)
                u0 * self.ziggurat.v / (self.ziggurat.f)(self.ziggurat.xs[0])
            } else {
                // z = U0 * xi
                u0 * self.ziggurat.xs[idx - 1]
            };

            // If z < x_{i+1}, accept z
            if z.abs() < self.ziggurat.xs[idx] {
                return z;
            }

            // Outside of inner rectangle, sample
            if idx == 0 {
                // Accept a z from tail using [Marsaglia 1964]
                let dmin = self.ziggurat.r;
                let mut x = self.rand_f64().ln() / dmin;
                let mut y = self.rand_f64().ln();
                while -2.0 * y < x * x {
                    x = self.rand_f64().ln() / dmin;
                    y = self.rand_f64().ln();
                }
                if z < 0.0 {
                    return x - dmin;
                } else {
                    return dmin - x;
                }
            } else {
                // If U1 * [ f(xi) - f(x_{i+1}) ] < f(z) - f(x_{i+1}), accept z
                let u1 = self.rand_f64();
                let fx = (self.ziggurat.f)(self.ziggurat.xs[idx - 1]);
                let fx1 = (self.ziggurat.f)(self.ziggurat.xs[idx]);
                let left = u1 * (fx - fx1);
                let right = (self.ziggurat.f)(z) - fx1;
                if left < right {
                    return z;
                }
            };
        }
    }

    #[cfg(feature = "num-complex")]
    pub fn crandn_f64(&mut self) -> Complex<f64> {
        Complex::new(self.randn_f64(), self.randn_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn randn_timing() {
        let n = 1024 * 1024 * 64;
        //let mut data = vec![0.0_f32; n];
        let mut awgn = Awgn::new(0, 512);
        let t0 = Instant::now();
        let data: Vec<f64> = (0..n).map(|_| awgn.randn_f64()).collect();
        let dt = t0.elapsed().as_secs_f64();
        println!("data[0]: {}", data[0]);
        println!("dt     : {}", dt);
        println!("MSPS   : {}", n as f64 / (1e6 * dt));
        assert_eq!(data.len(), n);
    }

    #[cfg(feature = "num-complex")]
    #[test]
    fn box_muller_timing() {
        let n = 1024 * 1024 * 64;
        //let mut data = vec![0.0_f32; n];
        let mut awgn = Awgn::new(0, 512);
        let t0 = Instant::now();
        //let data: Vec<_> = (0..n).map(|_| awgn.rand_f32()).collect();
        let data: Vec<_> = (0..n).map(|_| awgn.crandn_box_muller_f64()).collect();
        //let data: Vec<_> = awgn.rand_vec_f32(n);
        //awgn.fill_f32(&mut data);
        let dt = t0.elapsed().as_secs_f64();
        println!("data[0]: {}", data[0]);
        println!("dt     : {}", dt);
        println!("MSPS   : {}", n as f64 / (1e6 * dt));
        assert_eq!(data.len(), n);
    }

    #[test]
    fn test_find_r() {
        let (xc, r) = Ziggurat::find_r(128);
        println!("xc: {}, r: {}", xc, r);
        assert!(xc.abs() < 1e-5);
        assert!((r - 3.442619855899).abs() < 1e-9);
    }

    #[test]
    fn test_find_xs() {
        let c = 512;
        let xs = Ziggurat::find_xs(c);
        println!("xs: {:?}", xs);
        assert_eq!(xs.len(), c as usize);
    }

    #[cfg(feature = "num-complex")]
    #[test]
    fn ziggurat_timing() {
        let n = 1024 * 1024 * 64;
        //let mut data = vec![0.0_f32; n];
        let mut awgn = Awgn::new(0, 512);
        let t0 = Instant::now();
        //let data: Vec<_> = (0..n).map(|_| awgn.rand_f32()).collect();
        let data: Vec<_> = (0..n).map(|_| awgn.crandn_f64()).collect();
        //let data: Vec<_> = awgn.rand_vec_f32(n);
        //awgn.fill_f32(&mut data);
        let dt = t0.elapsed().as_secs_f64();
        println!("data[0]: {}", data[0]);
        println!("dt     : {}", dt);
        println!("MSPS   : {}", n as f64 / (1e6 * dt));
        assert_eq!(data.len(), n);
    }
}
