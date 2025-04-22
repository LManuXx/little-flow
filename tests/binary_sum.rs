// examples/binary_sum.rs
use littleflow::layer::dense::DenseLayer;
use littleflow::layer::trainable::{TrainableLayer, TrainableModel};
use littleflow::loss::mse::MeanSquaredError;
use littleflow::loss::Loss;
use littleflow::model::sequential::Sequential;
use littleflow::tensor::Tensor;
use littleflow::types::Accuracy;
use littleflow::layer::activation::{sigmoid, ActivationFn};

#[test]
fn main() {
    // Entradas: 4 bits (2 números de 2 bits)
    let inputs = vec![
        vec![0, 0, 0, 0], // 0 + 0 = 0
        vec![0, 0, 0, 1], // 0 + 1 = 1
        vec![0, 0, 1, 0], // 0 + 2 = 2
        vec![0, 0, 1, 1], // 0 + 3 = 3
        vec![0, 1, 0, 0], // 1 + 0 = 1
        vec![0, 1, 0, 1], // 1 + 1 = 2
        vec![0, 1, 1, 0], // 1 + 2 = 3
        vec![0, 1, 1, 1], // 1 + 3 = 4
        vec![1, 0, 0, 1], // 2 + 1 = 3
        vec![1, 1, 1, 1], // 3 + 3 = 6
    ];

    // Salidas: 3 bits (suma binaria)
    let targets = vec![
        vec![0, 0, 0],
        vec![0, 0, 1],
        vec![0, 1, 0],
        vec![0, 1, 1],
        vec![0, 0, 1],
        vec![0, 1, 0],
        vec![0, 1, 1],
        vec![1, 0, 0],
        vec![0, 1, 1],
        vec![1, 1, 0],
    ];

    // Conversion a Tensor<f32>
    let input_tensors: Vec<Tensor<f32>> = inputs
        .iter()
        .map(|v| Tensor::new(Accuracy::F32, v.iter().map(|&x| x as f32).collect(), vec![1, 4]))
        .collect();

    let target_tensors: Vec<Tensor<f32>> = targets
        .iter()
        .map(|v| Tensor::new(Accuracy::F32, v.iter().map(|&x| x as f32).collect(), vec![1, 3]))
        .collect();

    let mut model = Sequential::<f32>::new();
    model.add(DenseLayer::new(4, 8, Accuracy::F32)); // Capa oculta
    model.add(DenseLayer::new(8, 3, Accuracy::F32)); // Capa de salida

    let loss = MeanSquaredError;
    let activations: Vec<Option<ActivationFn<f32>>> = vec![Some(sigmoid), Some(sigmoid)];

    model.train(
        &input_tensors,
        &target_tensors,
        &loss,
        20000,
        0.1,
        &activations,
    );

    // Pruebas finales
    println!("\n--- Predicciones finales ---");
    for (i, input) in input_tensors.iter().enumerate() {
        let pred = model.forward(input, &activations);
        let data = pred.get_data().iter().map(|x| if *x > 0.5 { 1 } else { 0 }).collect::<Vec<_>>();
        println!("Input {:?} → Predicción: {:?} (Esperado: {:?})", inputs[i], data, targets[i]);
    }
}
