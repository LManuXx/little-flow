use std::ops::{Add, Div, Mul, Sub};

use crate::tensor::Tensor;
use crate::loss::Loss;

pub struct MeanSquaredError;

impl<T> Loss<T> for MeanSquaredError
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + From<f32> + Div<Output = T>,
{
    fn forward(&self, pred: &Tensor<T>, target: &Tensor<T>) -> Tensor<T> {
        if pred.get_shape() != target.get_shape() {
            panic!("Predicted and target tensors must have the same shape");
        }

        let error = pred.sub(target);
        let squared = error.mul_elementswise(&error);
        let sum = squared.sum_all();

        let n = T::from(pred.get_size() as f32);
        sum.scale(T::from(1.0f32) / n)
    }

    fn backward(&self, pred: &Tensor<T>, target: &Tensor<T>) -> Tensor<T> {
        if pred.get_shape() != target.get_shape() {
            panic!("Predicted and target tensors must have the same shape");
        }

        // grad = 2 * (pred - target) / n
        let error = pred.sub(target);
        let n = T::from(pred.get_size() as f32);
        let scale = T::from(2.0f32) / n;

        error.scale(scale)
    }
}
