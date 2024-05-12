
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Linear {
    pub weights: Tensor,
    prediction: Tensor
}

impl Linear {
    pub fn new(m: usize, n: usize) -> Linear {
        let weights = Tensor::zeros(m, n);
        let prediction = Tensor::zeros(m, 1);
        Linear { weights, prediction }
    } 

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Add unit row to x
        if (self.weights.n - 1) != x.m {
            panic!("Expected {} got {}", self.weights.n, x.m);
        }
        let ones = Tensor::ones(1, x.n);
        let x = x.concat(&ones);
        let product = self.weights.clone() * x; // TODO: don't want to be cloning..
        sigmoid(&product)
    }

}

fn sigmoid(y: &Tensor) -> Tensor {
    let e = 1.0_f64.exp();
    Tensor::singleton(1.0 / ( 1.0 + e.powf(y.data[0][0])))
}


