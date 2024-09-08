use core::panic;
use std::{cell::RefCell, rc::Rc};

use crate::{operations::Differentiable, tensor::Tensor};

pub trait Module {
    fn forward(&self, input: Tensor) -> Tensor;
    fn backward(&self, loss: Tensor);
    fn reset_grad(&self);
}

pub struct Linear {
    size: (usize, usize),
    pub weights: Rc<RefCell<Tensor>>,
    pub bias: Rc<RefCell<Tensor>>,
}

impl Linear {
    pub fn new(size: (usize, usize)) -> Linear {
        let (height, width) = size;
        if height <= 1 {
            panic!("Height for linear layer should be greater than 1")
        }
        let weights = Tensor::ones(height - 1, width).with_grad();
        let bias = Tensor::ones(1, width).with_grad();
        Linear {
            size: (height, width),
weights: Rc::new(RefCell::new(weights)), bias: Rc::new(RefCell::new(bias))
        }
    }

    pub fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}

impl Module for Linear {
    fn forward(&self, x: Tensor) -> Tensor {
        // Forward pass
        let weights = &*self.weights.borrow();
        let bias = &*self.bias.borrow();
        weights.set_grad(Tensor::singleton(0.0));
        bias.set_grad(Tensor::singleton(0.0));
        &(weights * &x) + bias
    }

    fn backward(&self, loss: Tensor) {
        loss.set_grad(Tensor::singleton(1.0));
        loss.backward();
        // let weight_update = learning_rate * self.weights.grad();
        // *self.weights -= &weight_update;
        // let bias_update = learning_rate * self.bias.grad();
        // *self.bias -= &bias_update;
    }

    fn reset_grad(&self) {
        self.weights.borrow().set_grad(Tensor::singleton(0.0));
        self.bias.borrow().set_grad(Tensor::singleton(0.0));
    }
}
