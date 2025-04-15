use crate::types::Accuracy;
use std::ops::{Add, Mul};

pub struct Tensor<T> {
    data: Vec<T>,      // Linear colection of values [1,2,3,4...]
    shape: Vec<usize>, // How the data is organized [2,3] 2 rows 3 columns
    size: usize,       // Product of shape 2x3 = 6 elements
    accuracy: Accuracy,
}

impl<T> Tensor<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    pub fn new(accuracy: Accuracy, data: Vec<T>, shape: Vec<usize>) -> Self {
        let size = shape.iter().product();

        if size != data.len() {
            panic!("Error: The number of elements doesn't match with Tensor shape");
        }

        Tensor {
            data,
            shape,
            accuracy,
            size,
        }
    }

    pub fn add(&self, other: &Tensor<T>) -> Tensor<T> {
        if self.shape != other.shape {
            panic!("Error: Cannot add Tensors with different shapes");
        }

        if self.accuracy != other.accuracy {
            panic!("Error: Cannot add Tensors with different accuracies");
        }

        let sum: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Tensor {
            data: sum,
            shape: self.shape.clone(),
            size: self.size,
            accuracy: self.accuracy,
        }
    }

    pub fn map<F>(&self, func: F) -> Tensor<T>
    where
        F: Fn(T) -> T,
    {
        let mapped_data: Vec<T> = self.data.iter().map(|&x| func(x)).collect();

        Tensor {
            data: mapped_data,
            shape: self.shape.clone(),
            size: self.size,
            accuracy: self.accuracy,
        }
    }

    pub fn matmul(&self, other: &Tensor<T>) -> Tensor<T> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            panic!("Error: Matmul only supports 2D tensors");
        }
        if self.shape[1] != other.shape[0] {
            panic!("Error: Shapes are not aligned for matrix multiplication");
        }

        let m = self.shape[0]; // filas de self
        let n = self.shape[1]; // columnas de self = filas de other
        let p = other.shape[1]; // columnas de other

        let mut result_data = vec![T::default(); m * p];

        // Multiplicación de matrices
        for i in 0..m {
            for j in 0..p {
                let mut sum = T::default();
                for k in 0..n {
                    let a = self.data[i * n + k];
                    let b = other.data[k * p + j];
                    sum = sum + (a * b);
                }
                result_data[i * p + j] = sum;
            }
        }

        Tensor {
            data: result_data,
            shape: vec![m, p],
            size: m * p,
            accuracy: self.accuracy, // Puedes decidir si copiar el accuracy de self o de other
        }
    }
}
