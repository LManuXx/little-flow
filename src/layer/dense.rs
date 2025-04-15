use crate::tensor::Tensor;

struct DenseLayer<T> {
    weights: Tensor<T>,
    bias: Tensor<T>
}