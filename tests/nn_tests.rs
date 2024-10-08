mod nn_tests {

    use approx::assert_relative_eq;
    use llm_rs::{
        data::TestData,
        nn::{Linear, Model, Module, ReLU},
        operations::Differentiable,
        optimizer::{Optimizer, StochasticGradientDescent},
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

        let layer = Linear::new(1, 1);
        let model = Model::new(vec![Box::new(layer)]);
        let optimizer = StochasticGradientDescent::new(learning_rate, model.parameters());

        let num_epochs = 500;
        for _ in 0..num_epochs {
            for sample in train.clone().into_iter() {
                model.reset_grad();
                // Forward pass
                let (x, y) = (sample.input, sample.output);
                let y_pred = model.forward(x);

                // Backward pass
                let loss = Differentiable::pow(&(y_pred - y.clone()), 2);
                model.backward(loss);

                // Weight update rule
                optimizer.step();
            }
        }

        let layer = model.layers[0].as_any().downcast_ref::<Linear>().unwrap();

        let (weights, bias) = (layer.weights.clone(), layer.bias.clone());
        let weights: &Tensor = &weights.borrow();
        let bias: &Tensor = &bias.borrow();
        println!("y = {}x + {}", weights.item(), bias.item());

        assert_relative_eq!(weights.item(), m, max_relative = 1e-5);
        assert_relative_eq!(bias.item(), b, max_relative = 1e-5);
    }

    fn create_data(x_1: i8, x_2: i8, y: i8) -> TestData {
        TestData {
            input: Tensor::from_vector(vec![vec![x_1.into(), x_2.into()]]),
            output: Tensor::singleton(y.into()),
        }
    }

    #[test]
    fn learn_xor() {
        let train: Vec<TestData> = vec![
            create_data(0, 0, 0),
            create_data(0, 1, 1),
            create_data(1, 0, 1),
            create_data(1, 1, 0),
        ];
        let learning_rate = 0.01;

        let model = Model::new(vec![
            Box::new(Linear::new(2, 4)),
            Box::new(ReLU {}),
            Box::new(Linear::new(4, 1)),
            Box::new(ReLU {}),
        ]);
        let optimizer = StochasticGradientDescent::new(learning_rate, model.parameters());

        let num_epochs = 500;
        for _ in 0..num_epochs {
            for sample in train.clone().into_iter() {
                model.reset_grad();
                // Forward pass
                let (x, y) = (sample.input, sample.output);
                let y_pred = model.forward(x);

                // Backward pass
                let loss = Differentiable::pow(&(y_pred - y.clone()), 2);
                println!("Loss: {}", loss.clone());
                model.backward(loss);

                // Weight update rule
                optimizer.step();
            }
        }

        for sample in train.clone().into_iter() {
            let (x, y) = (sample.input, sample.output);
            let prediction = model.forward(x);
            let (p, y) = (prediction.item(), y.item());
            println!("pred: {}, actual: {}", p, y)

            // assert_relative_eq!(prediction.item(), y.item(), max_relative = 1e-5);
        }
    }
}
