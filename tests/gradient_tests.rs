#[cfg(test)]
mod gradient_tests {
    use core::f64;

    use approx::assert_relative_eq;
    use llm_rs::{nn::TestData, operations::Differentiable, tensor::Tensor};

    #[test]
    fn simple_gradient_descent() {
        let m = 2.0;
        let range = 1..10;
        let train: Vec<TestData> = range
            .map(|x| {
                let x = x as f64;
                TestData {
                    input: Tensor::singleton(x).with_grad(),
                    output: Tensor::singleton(m * x).with_grad(),
                }
            })
            .collect();

        let weights = &mut Tensor::fill(1, 1, 1.0).with_grad();

        let learning_rate = 0.01;
        let num_epochs = 100;
        for _ in 0..num_epochs {
            // Forward pass
            for sample in train.clone().into_iter() {
                weights.set_grad(Tensor::singleton(0.0));
                let (x, y) = (sample.input, sample.output);
                let y_pred = &*weights * &x;
                let loss = &Differentiable::pow(&(y_pred - y.clone()), 2);

                loss.set_grad(Tensor::singleton(1.0));
                loss.backward(); // Backpropogate gradient

                let weight_update = learning_rate * weights.grad();
                *weights -= &weight_update;
            }
        }

        assert_eq!(weights.item(), m);
    }

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

        // TODO: add resetting grad to 0
        let weights = &mut Tensor::fill(1, 1, 1.0).with_grad();
        let bias = &mut Tensor::fill(1, 1, 1.0).with_grad();

        let learning_rate = 0.01;
        let num_epochs = 1000;
        for i in 0..num_epochs {
            // Forward pass
            let mut last_loss = Tensor::empty();
            for sample in train.clone().into_iter() {
                weights.set_grad(Tensor::singleton(0.0));
                bias.set_grad(Tensor::singleton(0.0));
                let (x, y) = (sample.input, sample.output);
                let y_pred = &(&*weights * &x) + bias;
                // println!("product: {}", y_pred);
                let y_pred_temp = &y_pred.clone();
                let loss = &Differentiable::pow(&(y_pred - y.clone()), 2);
                last_loss = loss.clone();
                loss.set_grad(Tensor::singleton(1.0));
                loss.backward(); // Backpropogate gradient

                // Weight update rule
                let weight_update = learning_rate * weights.grad();
                *weights -= &weight_update;
                let bias_update = learning_rate * bias.grad();
                *bias -= &bias_update;
            }
            if i % 100 == 0 {
                println!(
                    "epoch: {}, weights: {}, loss: {}",
                    i,
                    weights,
                    last_loss.clone()
                );
            }
        }
        println!("y = {}x + {}", weights.item(), bias.item());

        assert_relative_eq!(weights.item(), m, max_relative = 1e-5);
        assert_relative_eq!(bias.item(), b, max_relative = 1e-5);
    }

    #[test]
    fn relu_grad() {
        let a_val = 2.0;
        let b_val = -3.0;

        let a = Tensor::singleton(a_val).with_grad();
        let b = Tensor::singleton(b_val).with_grad();

        let c = a.relu();
        let d = b.relu();

        println!("{}, {}", a.last(), b.last());

        assert_eq!(2.0, c.item());
        assert_eq!(0.0, d.item());

        c.set_grad(Tensor::singleton(2.0));
        c.backward();
        d.set_grad(Tensor::singleton(2.0));
        d.backward();

        println!("{}, {}", a.grad(), b.grad());
        assert_eq!(2.0, a.grad().item());
        assert_eq!(0.0, b.grad().item());
    }

    #[test]
    fn small_computation_graph() {
        let a_val = 1.0;
        let b_val = 2.0;
        let c_val = 10.0;
        let f_val = -2.0;
        // y = f * ((a * b) + c)
        //   = f * (e + c)
        //   = f * d
        let a = Tensor::singleton(a_val).named("a".to_string()).with_grad();
        let b = Tensor::singleton(b_val).named("b".to_string()).with_grad();
        let e = &a * &b;
        let c = Tensor::singleton(c_val).named("c".to_string()).with_grad();
        let d = &e + &c;
        let f = Tensor::singleton(f_val).named("f".to_string()).with_grad();

        let y = &f * &d;

        // Assert correct value
        let y_val = y.item();
        assert_eq!(-24.0, y_val);
        let d_val = d.item();
        assert_eq!(12.0, d_val);
        let e_val = e.item();
        assert_eq!(2.0, e_val);

        // Assert correct last value
        let y_val = y.last().item();
        assert_eq!(-24.0, y_val);
        let d_val = d.last().item();
        assert_eq!(12.0, d_val);
        let e_val = e.last().item();
        assert_eq!(2.0, e_val);

        // Propogate gradient
        y.set_grad(Tensor::singleton(1.0));
        y.backward();
        // f = -2.0
        // d = e + c
        // ---------
        // y = f * d
        let y_grad = y.grad();
        assert_eq!(1.0, y_grad.item());
        // d.grad = dL/dd = (dL/dy)(dy/dd) = y.grad * f.last = 1 * -2 = -2
        let d_grad = d.grad();
        assert_eq!(d_grad, f.clone() * y.grad());
        assert_eq!(d_grad.item(), -2.0);
        // f.grad = dL/df = (dL/dy)(dy/df) = y.grad * d.last = 1 * 12 = 12
        let f_grad = f.grad();
        assert_eq!(f_grad, d.clone() * y.grad());
        assert_eq!(f_grad.item(), 12.0);

        // Assert correct gradient

        // c = 10.0
        // e = a * b
        // ---------
        // d = e + c
        // e.grad = dL/de = (dL/dd)(dd/de) = dL/dd * 1 = d.grad = -2
        assert_eq!(e.grad(), d.grad());
        assert_eq!(e.grad().item(), -2.0);
        // c.grad = dL/dc = (dL/dy)(dy/dd) = dL/dE * 1 = d.grad = -2
        assert_eq!(c.grad(), d.grad());
        assert_eq!(c.grad().item(), -2.0);

        // a = 1.0
        // b = 2.0
        // ---------
        // e = a * b
        // a.grad = dL/da = (dL/de)(de/da) = e.grad * b.last = -2 * 2 = -4
        assert_eq!(a.grad(), e.grad() * b.clone());
        assert_eq!(a.grad().item(), -4.0);
        // b.grad = dL/db = (dL/de)(de/db) = e.grad * a.last = -2 * 1 = -2
        assert_eq!(b.grad(), e.grad() * a.clone());
        assert_eq!(b.grad().item(), -2.0);
        //
    }
}
