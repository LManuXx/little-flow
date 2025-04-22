use crate::layer::trainable::{TrainableLayer, TrainableModel};
use crate::tensor::Tensor;
use crate::loss::Loss;

pub struct Sequential<T> {
    layers: Vec<Box<dyn TrainableLayer<T>>>,
}

impl<T> Sequential<T>
where
    T:'static + Copy + Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
{
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add(&mut self, layer: impl TrainableLayer<T> + 'static) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&self, input: &Tensor<T>, activations: &[Option<fn(T) -> T>]) -> Tensor<T> {
        let mut out = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            out = layer.forward(&out, activations[i]).unwrap();
        }
        out
    }

    pub fn predict_all(&self, inputs: &[Tensor<T>], activations: &[Option<fn(T) -> T>]) -> Vec<Tensor<T>> {
        inputs.iter().map(|x| self.forward(x, activations)).collect()
    }
}

impl<T, L> TrainableModel<T, L> for Sequential<T>
where
    T: 'static + Copy + Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::fmt::Debug,
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
    ) -> Vec<T> {
        let mut history = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut total_loss = T::default();

            for (input, target) in inputs.iter().zip(targets.iter()) {
                // FORWARD
                let mut activations_cache = vec![input.clone()];
                let mut output = input.clone();
                for (i, layer) in self.layers.iter().enumerate() {
                    output = layer.forward(&output, activations[i]).unwrap();
                    activations_cache.push(output.clone());
                }

                let loss = loss_fn.forward(&output, target);
                total_loss = total_loss + loss.get_data()[0];

                // BACKWARD
                let mut grad = loss_fn.backward(&output, target);
                for i in (0..self.layers.len()).rev() {
                    let (grad_input, grad_w, grad_b) = self.layers[i].backward(&activations_cache[i], &grad);
                    self.layers[i].update_params(&grad_w, &grad_b, learning_rate);
                    grad = grad_input;
                }
            }

            println!("Epoch {} - Loss: {:?}", epoch + 1, total_loss);
            history.push(total_loss);
        }

        history
    }
}
