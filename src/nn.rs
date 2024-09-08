use core::panic;
use std::{any::Any, cell::RefCell, rc::Rc};

use crate::{operations::Differentiable, tensor::Tensor};

pub trait Module {
    fn forward(&self, input: Tensor) -> Tensor;
    fn backward(&self, loss: Tensor) {
        loss.set_grad(Tensor::singleton(1.0));
        loss.backward();
    }
    fn reset_grad(&self);
    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>>;
    fn as_any(&self) -> &dyn Any;
}

pub struct Linear {
    size: (usize, usize),
    pub weights: Rc<RefCell<Tensor>>,
    pub bias: Rc<RefCell<Tensor>>,
}

impl Linear {
    pub fn new(size_in: usize, size_out: usize) -> Linear {
        let weights = Tensor::ones(size_in, size_out).with_grad();
        let bias = Tensor::ones(1, size_out).with_grad();
        Linear {
            size: (size_in, size_out),
            weights: Rc::new(RefCell::new(weights)),
            bias: Rc::new(RefCell::new(bias)),
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: Tensor) -> Tensor {
        // Forward pass
        let weights = &*self.weights.borrow();
        let bias = &*self.bias.borrow();
        weights.set_grad(Tensor::singleton(0.0));
        bias.set_grad(Tensor::singleton(0.0));
        println!("x: {}", x);
        println!("w: {}", weights);
        let a =&(&x * weights);
        println!("wx: {}", a);
        println!("b: {}", bias);
        let b = a + bias;
        println!("{}", a);
        b
    }

    fn reset_grad(&self) {
        self.weights.borrow().set_grad(Tensor::singleton(0.0));
        self.bias.borrow().set_grad(Tensor::singleton(0.0));
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![self.weights.clone(), self.bias.clone()]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct Model {
    pub layers: Vec<Box<dyn Module>>,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Model {
        Model { layers }
    }
}

impl Module for Model {
    fn forward(&self, input: Tensor) -> Tensor {
        let mut last_value = input;
        for layer in self.layers.iter() {
            let temp = last_value.clone();
            let (x, y) = temp.size;
            println!("size: ({}, {})", x, y);
            last_value = layer.forward(last_value);
        }
        last_value
    }

    fn reset_grad(&self) {
        for layer in self.layers.iter() {
            layer.reset_grad();
        }
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
