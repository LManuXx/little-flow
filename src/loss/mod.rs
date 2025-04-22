pub mod mse;

pub trait Loss<T> {
    fn forward(&self, pred: &crate::tensor::Tensor<T>, target: &crate::tensor::Tensor<T>) -> crate::tensor::Tensor<T>;
    fn backward(&self, pred: &crate::tensor::Tensor<T>, target: &crate::tensor::Tensor<T>) -> crate::tensor::Tensor<T>;
}
