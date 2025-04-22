// src/layer/trainable.rs

use crate::tensor::Tensor;

/// Trait para una capa entrenable individual (object-safe)
pub trait TrainableLayer<T>: 'static
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
{
    fn forward(&self, input: &Tensor<T>, activation: Option<fn(T) -> T>) -> Result<Tensor<T>, String>;
    fn backward(&self, input: &Tensor<T>, grad_output: &Tensor<T>) -> (Tensor<T>, Tensor<T>, Tensor<T>);
    fn update_params(&mut self, grad_w: &Tensor<T>, grad_b: &Tensor<T>, learning_rate: T);
}

/// Trait para modelos secuenciales completos (como Sequential)
use crate::loss::Loss;

pub trait TrainableModel<T, L>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
    L: Loss<T>,
{
    fn train(
        &mut self,
        inputs: &[Tensor<T>],
        targets: &[Tensor<T>],
        loss_fn: &L,
        epochs: usize,
        learning_rate: T,
        activations: &[Option<fn(T) -> T>],
    ) -> Vec<T>;
}
