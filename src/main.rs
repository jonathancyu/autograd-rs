mod tensor;
mod layers;

use std::vec;

use tensor::Tensor;
use layers::Linear;

fn main() {
    let (x, y) = get_data();
    let mut layer = Linear::new(1, 1);
    let num_epochs = 10;
    let lr = 0.5;

    println!("w {:?}", layer.weights.data);
    for epoch in 0..num_epochs {
        //for i in 0..(x.len()) {
        for i in 0..8 {
            let x_i = &x[i];
            let y_i = &y[i][0][0];

            let x_pred = layer.forward(x_i)[0][0];
            let x_sigm = sigmoid(x_pred);

            let d_k = y_i - x_sigm;
            for i in 0..layer.weights.n {
                let grad = lr * d_k * x_i[0][0] * derivative_sigmoid(x_pred);

                layer.weights[0][i] += lr * grad;
            }

            //layer.backward(&prediction, expected);
            println!("err {:?}", d_k);
            println!("w {:?}", layer.weights.data);
            //println!("epoch {}, pred: {:?}, exp: {:?}, err {:?}, weights {:?}", epoch, prediction, expected,error, layer.weights.data);
        }
    }
}
fn get_data() -> (Vec<Tensor>, Vec<Tensor>) {
    // TODO: return tensor
    let mut x: Vec<Tensor> = vec![];
    let mut y: Vec<Tensor> = vec![];

    for x_i in -100..100 {
        let x_i = x_i as f64;
        x.push(Tensor::singleton(x_i));
        //let y_i = f64::powi(x_i, 2) + x_i + 1.0;
        let y_i = 2.0*x_i + 1.0;
        y.push(Tensor::singleton(y_i));
    }

    (x, y)
}
fn derivative_sigmoid(val: f64) -> f64 {
    sigmoid(1.0 - val)
}

fn sigmoid(y: f64) -> f64{
    let e = 1.0_f64.exp();
    1.0 / ( 1.0 + e.powf(y))
}

