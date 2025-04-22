pub type ActivationFn<T> = fn(T) -> T;

pub fn relu<T>(x: T) -> T
where
    T: Copy + PartialOrd + Default,
{
    if x > T::default() { x } else { T::default() }
}

use num_traits::Float;

pub fn sigmoid<T>(x: T) -> T
where
    T: Float,
{
    let one: T = T::one();
    one / (one + (-x).exp())
}

pub fn tanh<T>(x: T) -> T
where
    T: Float,
{
    let one: T = T::one();
    (one.exp() - (-x).exp()) / (one.exp() + (-x).exp())
}
