use littleflow::{
    layer::dense::DenseLayer,
    layer::trainable::TrainableLayer,
    loss::mse::MeanSquaredError,
    loss::Loss,
    tensor::Tensor,
    types::Accuracy,
};

#[test]
fn main() {
    // Dataset: suma binaria simple
    let inputs = vec![
        Tensor::new(Accuracy::F32, vec![1.0, 0.0], vec![1, 2]),
        Tensor::new(Accuracy::F32, vec![0.0, 1.0], vec![1, 2]),
        Tensor::new(Accuracy::F32, vec![1.0, 1.0], vec![1, 2]),
        Tensor::new(Accuracy::F32, vec![0.0, 0.0], vec![1, 2]),
    ];

    let targets = vec![
        Tensor::new(Accuracy::F32, vec![1.0], vec![1, 1]),
        Tensor::new(Accuracy::F32, vec![1.0], vec![1, 1]),
        Tensor::new(Accuracy::F32, vec![2.0], vec![1, 1]),
        Tensor::new(Accuracy::F32, vec![0.0], vec![1, 1]),
    ];

    // Crea la capa densa
    let mut layer = DenseLayer::<f32>::new(2, 1, Accuracy::F32);
    let loss_fn = MeanSquaredError;

    // Entrenamiento
    let losses = layer.train(&inputs, &targets, &loss_fn, 100, 0.1, None);

    println!("\nHistorial de pérdida:");
    for (epoch, loss) in losses.iter().enumerate() {
        println!("Epoch {} → Loss: {:.4}", epoch + 1, loss);
    }

    // Predicciones
    let predictions = layer.predict_all(&inputs, None);
    println!("\nPredicciones finales:");
    for (i, pred) in predictions.iter().enumerate() {
        println!("Input {:?} → Predicción: {:?}", inputs[i].get_data(), pred.get_data());
    }
}
