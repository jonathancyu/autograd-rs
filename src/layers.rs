
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
        Linear { weights, prediction}
    } 

    pub fn forward(mut self, x: &Tensor) -> &Tensor {
        print!("{self:?}, {x:?}");
        // Add unit row to x
        if (self.weights.n - 1) != x.m {
            panic!("Expected {} got {}", self.weights.n, x.m);
        }
        let ones = Tensor::ones(1, x.n);
        let x = x.concat(&ones);
        self.prediction = self.weights * x;
        self.prediction
    }

}
