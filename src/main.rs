use llm_rs::{nn::TestData, operations::Differentiable, tensor::Tensor};

fn main() {
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
            // println!(
            //     "y_pred = {}, expected = {}, loss = {}",
            //     y_pred_temp, y, loss
            // );
            loss.set_grad(Tensor::singleton(1.0));
            loss.backward(); // Backpropogate gradient

            // Weight update rule
            // println!("w_grad = {}", weights.grad());
            let weight_update = learning_rate * weights.grad();
            *weights -= &weight_update;
            let bias_update = learning_rate * bias.grad();
            *bias -= &bias_update;
            // println!("weight_update = {}", weight_update);
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
    println!("y = {}x + {}", weights.item(), bias.item())
}
