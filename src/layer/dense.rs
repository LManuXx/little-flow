use std::ops::{Add, Mul};

use crate::{tensor::Tensor, types::Accuracy, types::Randomizable};

struct DenseLayer<T> {
    weights: Tensor<T>,
    bias: Tensor<T>,
}

impl<T> DenseLayer<T>
where
    T: Randomizable,
    T: Randomizable + Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    fn new(input_size: usize, output_size: usize, accuracy: Accuracy) -> DenseLayer<T> {
        let weight_data: Vec<T> = (0..input_size * output_size)
            .map(|_| T::random_weight())
            .collect();

        let bias_data: Vec<T> = (0..output_size).map(|_| T::default()).collect();

        let weights = Tensor::new(accuracy, weight_data, vec![input_size, output_size]);
        let bias = Tensor::new(accuracy, bias_data, vec![output_size]);

        DenseLayer { weights, bias }
    }
}
