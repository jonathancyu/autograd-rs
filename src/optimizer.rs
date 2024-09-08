use std::{cell::RefCell, rc::Rc};

use crate::{operations::Differentiable, tensor::Tensor};

pub trait Optimizer {
    fn step(&self);
}

pub struct StochasticGradientDescent {
    learning_rate: f64,
    parameters: Vec<Rc<RefCell<Tensor>>>,
}
impl StochasticGradientDescent {
    pub fn new(
        learning_rate: f64,
        parameters: Vec<Rc<RefCell<Tensor>>>,
    ) -> StochasticGradientDescent {
        StochasticGradientDescent {
            learning_rate,
            parameters,
        }
    }
}

impl Optimizer for StochasticGradientDescent {
    fn step(&self) {
        self.parameters.clone().into_iter().for_each(|parameter| {
            let mut parameter = parameter.borrow_mut();
            let weight_update = self.learning_rate * parameter.grad();
            *parameter -= &weight_update;
        });
    }
}
