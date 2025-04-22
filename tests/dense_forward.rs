use littleflow::layer::dense::DenseLayer;
use littleflow::layer::activation::relu;
use littleflow::types::Accuracy;
use littleflow::tensor::Tensor;

#[test]
fn dense_layer_forward_with_relu() {
    let layer = DenseLayer::<f32>::new(3, 2, Accuracy::F32);
    let input_data = vec![1.0, -2.0, 0.5];
    let input = Tensor::new(Accuracy::F32, input_data, vec![1, 3]);

    let result = layer.forward(&input, Some(relu)).unwrap();

    assert_eq!(result.get_shape(), &[1, 2]);
    for &x in result.get_data() {
        assert!(x >= 0.0); // porque aplicamos relu
    }
}
