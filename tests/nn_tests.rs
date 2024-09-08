mod nn_tests {

    use approx::assert_relative_eq;
    use llm_rs::{
        data::TestData,
        nn::{Linear, Module, Optimizer, StochasticGradientDescent},
        operations::Differentiable,
        tensor::Tensor,
    };

    #[test]
    fn learn_linear_equation() {
        let m = -3.0;
        let b = 13.0;
        let range = 1..10;
        let train: Vec<TestData> = range
            .map(|x| {
                let x = x as f64;
                TestData {
                    input: Tensor::singleton(x).with_grad(),
                    output: Tensor::singleton(m * x + b).with_grad(),
                }
            })
            .collect();
        let learning_rate = 0.01;

        let layer = Linear::new((2, 1));
        let optimizer = StochasticGradientDescent::new(learning_rate, layer.parameters());

        let num_epochs = 500;
        for _ in 0..num_epochs {
            for sample in train.clone().into_iter() {
                layer.reset_grad();
                // Forward pass
                let (x, y) = (sample.input, sample.output);
                let y_pred = layer.forward(x);

                // Backward pass
                let loss = Differentiable::pow(&(y_pred - y.clone()), 2);
                layer.backward(loss);


                // Weight update rule
                optimizer.step();
            }
        }

        let (weights, bias) = (layer.weights.clone(), layer.bias.clone());
        let weights: &Tensor = &weights.borrow();
        let bias: &Tensor = &bias.borrow();
        println!("y = {}x + {}", weights.item(), bias.item());

        assert_relative_eq!(weights.item(), m, max_relative = 1e-5);
        assert_relative_eq!(bias.item(), b, max_relative = 1e-5);
    }
}
