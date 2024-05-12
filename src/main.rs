mod tensor;
mod layers;

use std::vec;

use tensor::Tensor;
use layers::Linear;

fn main() {
    // TODO: 
    let a = Tensor::fill(2, 2, 2.0);
    let b = Tensor::fill(2, 3, 1.0);
    println!("{:?} {:?}", a, b);
    println!("{:?}", a * b);

    let (x, y) = get_data();
    let layer = Linear::new(1, 2);
    let num_epochs = 1;
    for _epoch in 0..num_epochs {
        for i in 0..(x.len()) {
            let prediction = layer.forward(&x[i]);
            let expected = &y[i];
            let _error = prediction.data[0][0] - expected.data[0][0];
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
