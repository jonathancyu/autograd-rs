use crate::tensor::Tensor;

#[derive(Clone)]
pub struct TestData {
    pub input: Tensor,
    pub output: Tensor,
}

pub trait Layer {
    fn apply(&self, input: Tensor) -> Tensor;
}

pub struct Linear {
    size: (usize, usize),
    weights: Tensor,
    bias: Tensor,
}

impl Layer for Linear {
    fn apply(&self, input: Tensor) -> Tensor {
        todo!()
    }
}
