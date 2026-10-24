//! CPU mirror of the WGSL u64 emulation of [`super::hash_u`].
//!
//! WGSL has no 64-bit integers, so the GPU port of the value-noise hash works on `(hi, lo)` pairs of
//! `u32`. The functions here use exactly the operations the shader uses (32-bit wrapping mul, xor,
//! shifts, a 16-bit-limb high multiply), so a bit-exact match with `hash_u` proves the GPU
//! arithmetic without a GPU. Test-only: the product code path stays the u64 reference.

/// A 64-bit value as two `u32` halves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct U64 {
    pub hi: u32,
    pub lo: u32,
}

impl U64 {
    pub(crate) fn from_u64(v: u64) -> Self {
        U64 { hi: (v >> 32) as u32, lo: v as u32 }
    }

    pub(crate) fn to_u64(self) -> u64 {
        ((self.hi as u64) << 32) | self.lo as u64
    }

    /// Sign-extends an `i32` to 64 bits (the shader's `u64_of_i32`).
    pub(crate) fn from_i32(v: i32) -> Self {
        U64 { hi: if v < 0 { u32::MAX } else { 0 }, lo: v as u32 }
    }

    pub(crate) fn xor(self, o: U64) -> U64 {
        U64 { hi: self.hi ^ o.hi, lo: self.lo ^ o.lo }
    }

    /// Logical right shift for `0 < n <= 32` (the shader only uses 29, 31 and 32).
    pub(crate) fn shr(self, n: u32) -> U64 {
        if n >= 32 { U64 { hi: 0, lo: self.hi >> (n - 32) } } else { U64 { hi: self.hi >> n, lo: (self.lo >> n) | (self.hi << (32 - n)) } }
    }

    /// Wrapping 64-bit multiply, built from 32-bit products.
    pub(crate) fn mul(self, o: U64) -> U64 {
        let lo = self.lo.wrapping_mul(o.lo);
        let hi = mulhi32(self.lo, o.lo).wrapping_add(self.lo.wrapping_mul(o.hi)).wrapping_add(self.hi.wrapping_mul(o.lo));
        U64 { hi, lo }
    }
}

/// High 32 bits of the full 32×32 product, from 16-bit limbs (no 64-bit intermediate).
pub(crate) fn mulhi32(a: u32, b: u32) -> u32 {
    let (ah, al, bh, bl) = (a >> 16, a & 0xffff, b >> 16, b & 0xffff);
    let t = al.wrapping_mul(bl);
    let w1 = ah.wrapping_mul(bl).wrapping_add(t >> 16);
    let w2 = al.wrapping_mul(bh).wrapping_add(w1 & 0xffff);
    ah.wrapping_mul(bh).wrapping_add(w1 >> 16).wrapping_add(w2 >> 16)
}

/// The shader's `hash_u`: same constants and steps as [`super::hash_u`], on `U64` pairs.
pub(crate) fn hash_u_pair(x: i64, y: i64, z: i64, seed: u64) -> U64 {
    let c1 = U64::from_u64(0x9E37_79B9_7F4A_7C15);
    let c2 = U64::from_u64(0xC2B2_AE3D_27D4_EB4F);
    let c3 = U64::from_u64(0x1656_67B1_9E37_79F9);
    let c4 = U64::from_u64(0x27D4_EB2F_1656_67C5);
    let c5 = U64::from_u64(0xBF58_476D_1CE4_E5B9);
    let c6 = U64::from_u64(0x94D0_49BB_1331_11EB);
    let (xv, yv, zv) = (U64::from_u64(x as u64), U64::from_u64(y as u64), U64::from_u64(z as u64));
    let mut h = xv.mul(c1).xor(yv.mul(c2)).xor(zv.mul(c3)).xor(U64::from_u64(seed).mul(c4));
    h = h.xor(h.shr(31));
    h = h.mul(c5);
    h = h.xor(h.shr(29));
    h = h.mul(c6);
    h.xor(h.shr(32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfx::hash_u;

    /// xorshift64*: a deterministic stream of inputs for the comparison.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    #[test]
    fn mulhi32_matches_wide_product() {
        let mut r = Rng(0x1234_5678_9ABC_DEF1);
        for _ in 0..200_000 {
            let (a, b) = (r.next() as u32, r.next() as u32);
            assert_eq!(mulhi32(a, b), ((a as u64 * b as u64) >> 32) as u32, "a={a:#x} b={b:#x}");
        }
        assert_eq!(mulhi32(u32::MAX, u32::MAX), 0xFFFF_FFFE);
    }

    #[test]
    fn u64_mul_matches_wrapping_mul() {
        let mut r = Rng(0xDEAD_BEEF_0BAD_F00D);
        for _ in 0..200_000 {
            let (a, b) = (r.next(), r.next());
            assert_eq!(U64::from_u64(a).mul(U64::from_u64(b)).to_u64(), a.wrapping_mul(b));
        }
    }

    #[test]
    fn hash_pair_matches_hash_u_bit_exact() {
        // Edge values first, then a few million deterministic random inputs.
        let edges = [0i64, 1, -1, 2, -2, i64::MIN, i64::MAX, 0x7FFF_FFFF, -0x8000_0000];
        let seeds = [0u64, 1, u64::MAX, 0x8000_0000_0000_0000, 42];
        for &x in &edges {
            for &y in &edges {
                for &z in &edges[..4] {
                    for &s in &seeds {
                        assert_eq!(hash_u_pair(x, y, z, s).to_u64(), hash_u(x, y, z, s), "x={x} y={y} z={z} s={s}");
                    }
                }
            }
        }
        let mut r = Rng(0x9E37_0000_0000_0001);
        for i in 0..4_000_000u32 {
            let (x, y, z, s) = (r.next() as i64, r.next() as i64, r.next() as i64, r.next());
            // Grid cells are i32 in the shader; cover that range densely as well.
            let (xi, yi) = (x as i32 as i64, y as i32 as i64);
            assert_eq!(hash_u_pair(xi, yi, z, s).to_u64(), hash_u(xi, yi, z, s), "iter {i}");
        }
    }

    #[test]
    fn from_i32_sign_extends_like_i64() {
        for v in [0i32, 1, -1, i32::MIN, i32::MAX, -12345] {
            assert_eq!(U64::from_i32(v).to_u64(), v as i64 as u64);
        }
    }
}
