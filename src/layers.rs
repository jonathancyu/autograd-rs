use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Linear {
    pub weights: Tensor,
}

impl Linear {
    pub fn new(m: usize, n: usize) -> Linear {
        let weights = Tensor::zeros(m, n);
        Linear { weights }
    } 

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Add unit row to x
        if (self.weights.n - 1) != x.m {
            panic!("Expected {} got {}", self.weights.n-1, x.m);
        }
        let ones = Tensor::ones(1, x.n);
        let x = x.concat(&ones);

        self.weights.clone() * x
        //(self.weights.clone() * x).sigmoid()
    }



    //pub fn backward(&mut self, x: &Tensor, : &Tensor, lr: f64) {
    //
    //}

}

