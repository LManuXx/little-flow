use half::f16;
use rand::Rng;

#[derive(PartialEq, Copy, Clone)]
pub enum Accuracy {
    U8,
    I8,
    F16,
    F32,
}

pub trait Randomizable {
    fn random_weight() -> Self;
}

impl Randomizable for u8 {
    fn random_weight() -> Self {
        let mut rng = rand::rng();
        let val = rng.random_range(-0.1f32..=0.1);
        ((val * 127.0) + 128.0).clamp(0.0, 255.0) as u8
    }
}

impl Randomizable for i8 {
    fn random_weight() -> Self {
        let mut rng = rand::rng();
        let val = rng.random_range(-0.1f32..=0.1);
        (val * 127.0).clamp(-128.0, 127.0) as i8
    }
}

impl Randomizable for f16 {
    fn random_weight() -> Self {
        let mut rng = rand::rng();
        let val = rng.random_range(-0.1f32..=0.1);
        f16::from_f32(val)
    }
}

impl Randomizable for f32 {
    fn random_weight() -> Self {
        let mut rng = rand::rng();
        rng.random_range(-0.1..=0.1)
    }
}
