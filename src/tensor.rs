use crate::types::Accuracy;
use std::ops::{Add, Mul, Sub};
#[derive(Clone)]
pub struct Tensor<T> {
    data: Vec<T>,      // Linear colection of values [1,2,3,4...]
    shape: Vec<usize>, // How the data is organized [2,3] 2 rows 3 columns
    size: usize,       // Product of shape 2x3 = 6 elements
    accuracy: Accuracy,
}

impl<T> Tensor<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default + Sub<Output = T>,
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
        if self.accuracy != other.accuracy {
            panic!("Error: Cannot add Tensors with different accuracies");
        }
    
        // Case 1: shapes are exactly equal
        if self.shape == other.shape {
            let sum: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect();
    
            return Tensor {
                data: sum,
                shape: self.shape.clone(),
                size: self.size,
                accuracy: self.accuracy,
            };
        }
    
        // Case 2: broadcasting case: [batch_size, output_size] + [output_size]
        if self.shape.len() == 2 && other.shape.len() == 1 {
            let batch_size = self.shape[0];
            let output_size = self.shape[1];
    
            if other.shape[0] != output_size {
                panic!("Error: Cannot broadcast: bias shape does not match output size");
            }
    
            let mut result_data = Vec::with_capacity(self.data.len());
    
            for i in 0..batch_size {
                for j in 0..output_size {
                    let a = self.data[i * output_size + j];
                    let b = other.data[j];
                    result_data.push(a + b);
                }
            }
    
            return Tensor {
                data: result_data,
                shape: self.shape.clone(),
                size: self.size,
                accuracy: self.accuracy,
            };
        }
    
        // Fallback: incompatible shapes
        panic!("Error: Cannot add Tensors with incompatible shapes");
    }

    pub fn sum(&self, axis: usize) -> Tensor<T>
where
    T: Copy + Default + Add<Output = T>,
{
    if self.shape.len() != 2 {
        panic!("sum() currently only supports 2D tensors");
    }

    let rows = self.shape[0];
    let cols = self.shape[1];

    if axis > 1 {
        panic!("Invalid axis: {} (only 0 or 1 are supported)", axis);
    }

    let (output_len, output_shape) = if axis == 0 {
        (cols, vec![cols])
    } else {
        (rows, vec![rows])
    };

    let mut result = vec![T::default(); output_len];

    match axis {
        // sum over rows (axis 0): result shape [cols]
        0 => {
            for row in 0..rows {
                for col in 0..cols {
                    let idx = row * cols + col;
                    result[col] = result[col] + self.data[idx];
                }
            }
        }

        // sum over cols (axis 1): result shape [rows]
        1 => {
            for row in 0..rows {
                for col in 0..cols {
                    let idx = row * cols + col;
                    result[row] = result[row] + self.data[idx];
                }
            }
        }

        _ => unreachable!(),
    }

    Tensor {
        data: result,
        shape: output_shape,
        size: output_len,
        accuracy: self.accuracy,
    }
}

pub fn sum_all(&self) -> Tensor<T>
where
    T: Copy + Add<Output = T> + Default,
{
    let mut total = T::default();
    for &x in &self.data {
        total = total + x;
    }

    Tensor {
        data: vec![total],
        shape: vec![1],
        size: 1,
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

    pub fn sub(&self, other: &Tensor<T>) -> Tensor<T> {
        if self.accuracy != other.accuracy {
            panic!("Error: Cannot subtract Tensors with different accuracies");
        }

        // Case 1: shapes are exactly equal
        if self.shape == other.shape {
            let diff: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a - b)
                .collect();

            return Tensor {
                data: diff,
                shape: self.shape.clone(),
                size: self.size,
                accuracy: self.accuracy,
            };
        }

        // Fallback: incompatible shapes
        panic!("Error: Cannot subtract Tensors with incompatible shapes");
    }

    pub fn mul_elementswise(&self, other: &Tensor<T>) -> Tensor<T> {
        if self.accuracy != other.accuracy {
            panic!("Error: Cannot mult Tensors with different accuracies");
        }

        // Case 1: shapes are exactly equal
        if self.shape == other.shape {
            let diff: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a * b)
                .collect();

            return Tensor {
                data: diff,
                shape: self.shape.clone(),
                size: self.size,
                accuracy: self.accuracy,
            };
        }

        // Fallback: incompatible shapes
        panic!("Error: Cannot mult Tensors with incompatible shapes");
    }

    pub fn transpose(&self) -> Tensor<T> {
        if self.shape.len() != 2 {
            panic!("Error: Transpose only supports 2D tensors");
        }

        let m = self.shape[0]; // filas de self
        let n = self.shape[1]; // columnas de self

        let mut transposed_data = vec![T::default(); m * n];

        // Transposición de matrices
        for i in 0..m {
            for j in 0..n {
                transposed_data[j * m + i] = self.data[i * n + j];
            }
        }

        Tensor {
            data: transposed_data,
            shape: vec![n, m],
            size: m * n,
            accuracy: self.accuracy,
        }
    }

    pub fn scale(&self, fact: T) -> Tensor<T> {
        
        let scaled_data: Vec<T> = self.get_data()
            .iter()
            .map(|&a| a*fact)
            .collect();

        Tensor { data: scaled_data, shape: self.shape.clone(), size: self.size, accuracy: self.accuracy }
    }

    pub fn get_shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn get_data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    pub fn get_accuracy(&self) -> &Accuracy {
        &self.accuracy
    }
}
