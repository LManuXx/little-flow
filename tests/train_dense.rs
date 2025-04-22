use littleflow::{
    layer::dense::DenseLayer,
    loss::mse::MeanSquaredError,
    loss::Loss,
    tensor::Tensor,
    types::Accuracy,
};

#[test]
pub fn train_dense_layer() {
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


    let mut layer = DenseLayer::<f32>::new(2, 1, Accuracy::F32);
    let loss_fn = MeanSquaredError;

    let learning_rate = 0.1;
    let epochs = 100;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let pred = layer.forward(input, None).unwrap();
            let loss = loss_fn.forward(&pred, target);
            total_loss += loss.get_data()[0];

            let dL_dy = loss_fn.backward(&pred, target);
            let (_grad_input, grad_weights, grad_bias) = layer.backward(input, &dL_dy);

            // Actualización de parámetros
            layer.set_weights(&layer.get_weights().sub(&grad_weights.scale(learning_rate)));
            layer.update_bias(&layer.get_bias().sub(&grad_bias.scale(learning_rate)));
        }

        println!("Epoch {} - Loss: {:.5}", epoch + 1, total_loss / inputs.len() as f32);
    }

    // Probar resultado final
    let test = Tensor::new(Accuracy::F32, vec![1.0, 1.0], vec![1, 2]);
    let prediction = layer.forward(&test, None).unwrap();
    println!("Final prediction for [1.0, 1.0]: {:?}", prediction.get_data());
}
