use crate::tensor::Tensor;

#[derive(Clone)]
pub struct TestData {
    pub input: Tensor,
    pub output: Tensor,
}
