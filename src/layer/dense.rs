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

    pub fn backward(
        &self,
        input: &Tensor<T>,
        dL_dy: &Tensor<T>,
    ) -> (Tensor<T>, Tensor<T>, Tensor<T>)
    where
        T: Copy
            + Default
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + From<f32>,
    {
        // 1. Gradiente respecto a los pesos: Xᵗ * dL_dy
        let input_T = input.transpose();
        let grad_weights = input_T.matmul(dL_dy);
    
        // 2. Gradiente respecto al bias: sum(delta) sobre axis 0
        let grad_bias = dL_dy.sum(0);
    
        // 3. Gradiente respecto al input: dL_dy * Wᵗ
        let weights_T = self.weights.transpose();
        let grad_input = dL_dy.matmul(&weights_T);
    
        (grad_input, grad_weights, grad_bias)
    }

    pub fn get_weights(&self) -> &Tensor<T> {
        &self.weights
    }

    pub fn get_bias(&self) -> &Tensor<T> {
        &self.bias
    }
    
    pub fn set_weights(&mut self, new_weights: &Tensor<T>) {
        if new_weights.get_shape() != self.weights.get_shape() {
            panic!("New weights shape does not match the current weights shape");
        }
        self.weights = (*new_weights).clone();
    }

    pub fn update_bias(&mut self, new_bias: &Tensor<T>) {
        if new_bias.get_shape() != self.bias.get_shape() {
            panic!("New bias shape does not match the current bias shape");
        }
        self.bias = (*new_bias).clone();
    }
    
}
