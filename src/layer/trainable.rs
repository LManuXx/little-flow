use crate::tensor::Tensor;
use crate::loss::Loss;

pub trait TrainableLayer<T>
where
    T: Copy
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::fmt::Debug,
{
    fn forward(&self, input: &Tensor<T>, activation: Option<fn(T) -> T>) -> Result<Tensor<T>, String>;
    fn backward(&self, input: &Tensor<T>, grad_output: &Tensor<T>) -> (Tensor<T>, Tensor<T>, Tensor<T>);
    fn update_params(&mut self, grad_w: &Tensor<T>, grad_b: &Tensor<T>, learning_rate: T);

    /// Entrena el modelo y devuelve un historial de pérdida (loss) por epoch
    fn train<L: Loss<T>>(
        &mut self,
        inputs: &[Tensor<T>],
        targets: &[Tensor<T>],
        loss_fn: &L,
        epochs: usize,
        learning_rate: T,
        activation: Option<fn(T) -> T>,
    ) -> Vec<T> {
        let mut history = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut total_loss = T::default();

            for (input, target) in inputs.iter().zip(targets.iter()) {
                let pred = self.forward(input, activation).unwrap();
                let loss = loss_fn.forward(&pred, target);
                let grad = loss_fn.backward(&pred, target);

                let (_, grad_weights, grad_bias) = self.backward(input, &grad);
                self.update_params(&grad_weights, &grad_bias, learning_rate);

                total_loss = total_loss + loss.get_data()[0];
            }

            println!("Epoch {} - Loss: {:?}", epoch + 1, total_loss);
            history.push(total_loss);
        }

        history
    }

    /// Aplica forward a todos los inputs y devuelve las predicciones
    fn predict_all(
        &self,
        inputs: &[Tensor<T>],
        activation: Option<fn(T) -> T>,
    ) -> Vec<Tensor<T>> {
        inputs
            .iter()
            .map(|input| self.forward(input, activation).unwrap())
            .collect()
    }

    fn predict(
        &self,
        input: &Tensor<T>,
        activation: Option<fn(T) -> T>,
    ) -> Result<Tensor<T>, String> {
        self.forward(input, activation)
    }    
}
