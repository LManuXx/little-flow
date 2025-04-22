use std::ops::{Add, Mul, Sub};

use crate::{
    tensor::Tensor,
    types::{Accuracy, Randomizable},
};

use crate::layer::activation::ActivationFn;

pub struct DenseLayer<T> {
    weights: Tensor<T>,
    bias: Tensor<T>,
}

impl<T> DenseLayer<T>
where
    T: Randomizable + Copy + Add<Output = T> + Mul<Output = T> + Default + Sub<Output = T>,
{
    pub fn new(input_size: usize, output_size: usize, accuracy: Accuracy) -> DenseLayer<T> {
        let weight_data: Vec<T> = (0..input_size * output_size)
            .map(|_| T::random_weight())
            .collect();

        let bias_data: Vec<T> = (0..output_size).map(|_| T::default()).collect();

        let weights = Tensor::new(accuracy, weight_data, vec![input_size, output_size]);
        let bias = Tensor::new(accuracy, bias_data, vec![output_size]);

        DenseLayer { weights, bias }
    }

    pub fn forward(
        &self,
        input: &Tensor<T>,
        activation: Option<ActivationFn<T>>,
    ) -> Result<Tensor<T>, String> {
        if input.get_shape().len() != 2 {
            return Err("Input tensor must be 2D".into());
        }

        if input.get_shape()[1] != self.weights.get_shape()[0] {
            return Err("Input tensor size does not match weight matrix".into());
        }

        if self.bias.get_shape().len() != 1
            || self.bias.get_shape()[0] != self.weights.get_shape()[1]
        {
            return Err("Bias shape is not compatible with output shape".into());
        }

        let linear_output = input.matmul(&self.weights);

        let output = linear_output.add(&self.bias);

        match activation {
            Some(activation_fn) => {
                let activated_output = output.map(activation_fn);
                Ok(activated_output)
            }
            None => Ok(output),
        }
    }
}
