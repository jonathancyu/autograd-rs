use std::{
    cell::RefCell,
    clone,
    ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign},
    rc::Rc,
};

use crate::tensor::Tensor;

#[derive(Clone)]
pub struct Gradient {
    pub operation: GradientOperation,
    pub last: Option<Tensor>,
    pub value: Option<Tensor>, // Shouldn't grad be ties to operation?
}

fn format_name(tensor: &Tensor) -> String {
    if !tensor.name.is_empty() {
        return tensor.name.clone();
    }
    tensor.to_string()
}

fn unary_label(operation: String, tensor: &Tensor) -> String {
    let tensor = format_name(tensor);
    format!("({} {})", operation, tensor)
}

fn binary_label(left: &Tensor, operation: String, right: &Tensor) -> String {
    let left = format_name(left);
    let right = format_name(right);
    format!("({} {} {})", left, operation, right)
}

impl Default for Gradient {
    fn default() -> Self {
        Gradient {
            operation: GradientOperation::None,
            last: None,
            value: None,
        }
    }
}

impl Gradient {
    pub fn wrap(self) -> Rc<RefCell<Gradient>> {
        Rc::new(RefCell::new(self))
    }
}

#[derive(Clone)]
pub enum Parents {
    None,
    Unary(Rc<RefCell<Gradient>>),
}

// TODO: no words
#[derive(Debug, Clone)]
pub enum GradientOperation {
    None,
    Neg(Tensor),
    ReLU(Tensor),
    Pow(Tensor, i32),
    Mean(Tensor),
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
}

pub trait Differentiable {
    fn grad(&self) -> Tensor;
    fn with_grad(self) -> Self;
    fn set_grad(&self, grad: Tensor);
    fn reset_grad(&self);
    fn add_grad(&self, grad: Tensor);

    fn last(&self) -> Tensor;

    fn backward(&self);

    // TODO: move these elsewhere
    fn relu(&self) -> Tensor;
    fn mean(&self) -> Tensor;
    fn pow(&self, exp: i32) -> Tensor;
}

impl Differentiable for Tensor {
    fn with_grad(self) -> Self {
        let mut gradient = self.gradient.borrow_mut();
        match gradient.value {
            Some(_) => println!("Tensor already has grad enabled."),
            None => {
                let (m, n) = self.size;
                gradient.value = Some(Tensor::zeros(m, n));
                gradient.last = Some(Tensor::from_vector(self.data.clone()))
            }
        };
        self.clone() // TODO: is this bad?
    }
    fn grad(&self) -> Tensor {
        let gradient = self.gradient.borrow();
        let value = gradient
            .value
            .clone()
            .expect("Tensor doesn't have grad enabled");
        value.clone()
    }

    fn reset_grad(&self) {
        let (w, h) = self.size;
        self.set_grad(Tensor::zeros(w, h));
    }

    fn set_grad(&self, grad: Tensor) {
        let mut gradient = self.gradient.borrow_mut();
        gradient.value = Some(grad)
    }

    fn add_grad(&self, grad: Tensor) {
        let mut gradient = self.gradient.borrow_mut();
        if let Some(value) = gradient.value.clone() {
            gradient.value = Some(value + grad);
        }
        // TODO: should this fail silently? I think so, b/c should do nothing with tensors w/o grad
        // enabled, right?
        // What if grad isn't enabled but gradients flow through this node?
    }

    fn last(&self) -> Tensor {
        let gradient = self.gradient.borrow();
        match &gradient.last {
            Some(value) => value.clone(),
            None => self.clone(),
        }
    }

    fn backward(&self) {
        let grad = self.grad();
        let gradient = self.gradient.borrow();
        let g_debug = gradient.clone();
        println!(
            "BACKWARD: {:?} \t\t = {}, grad = {}",
            g_debug.operation,
            g_debug.last.unwrap(),
            g_debug.value.unwrap()
        );
        match &gradient.operation {
            GradientOperation::None => {}
            GradientOperation::Neg(a) => {
                // y = -a
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * -1
                a.add_grad(-grad.clone());
                a.backward();
                // println!("{}: {}", a.name, a.grad());
            }
            GradientOperation::Add(a, b) => {
                // y = a + b
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * 1
                // b.grad = dL/db = (dL/dy)(dy/db) = grad * 1
                a.add_grad(grad.clone());
                b.add_grad(grad.clone());
                a.backward();
                b.backward();
                // println!("{}: {}, {}: {}", a.name, a.grad(), b.name, b.grad());
            }
            GradientOperation::Sub(a, b) => {
                // y = a - b
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * 1
                // b.grad = dL/db = (dL/dy)(dy/db) = grad * -1
                a.add_grad(grad.clone());
                b.add_grad(-grad.clone());
                a.backward();
                b.backward();
                // println!("{}: {}, {}: {}", a.name, a.grad(), b.name, b.grad());
            }
            GradientOperation::Mul(a, b) => {
                // y = a * b
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * b
                // b.grad = dL/db = (dL/dy)(dy/db) = grad * a
                let a_last = a.last();
                let b_last = b.last();
                let (a1, a2) = a_last.size;
                let (b1, b2) = b_last.size;
                println!("a_size: {}x{}, b_size: {}x{}", a1, a2, b1, b2);
                let (g1, g2) = grad.size;
                println!("grad size: {}x{}", g1, g2);
                let a_partial = grad.clone() * b_last.transpose();
                println!("a_partial: {}", a_partial.clone());
                a.add_grad(a_partial);
                let b_partial = (grad.clone() * a_last).transpose();
                println!("b_partial: {}", b_partial.clone());
                b.add_grad(b_partial);
                a.backward();
                b.backward();
            }
            GradientOperation::ReLU(a) => {
                // y = [ x >= 0: x, x < 0: 0 ]
                // dy/dx = [x >= 0: 1, x < 0: 0]
                let a_last = a.last();
                a.add_grad(
                    a_last.apply(|i, j, last| if last[i][j] >= 0.0 { grad[i][j] } else { 0.0 }),
                );
                // println!("{}: {}", a.name, a.grad());
                a.backward();
            }
            GradientOperation::Pow(a, b) => {
                // y = a^b
                // dy/da = ba^(b-1)
                let a_last = a.last();
                a.add_grad(
                    a_last.apply(|i, j, last| (*b as f64) * last[i][j].powf((b - 1) as f64)),
                );
                // println!("{}: {}, b: {}", a.name, a.grad(), b);
                a.backward();
            }
            GradientOperation::Mean(a) => {
                // y = mean(a)
                // dy/da = ba^(b-1)
                let a_last = a.last();
                let denominator = a_last.num_elements() as f64;
                a.add_grad(a_last.apply(|i, j, last| last[i][j] / denominator));
                // println!("{}: {}", a.name, a.grad());
                a.backward();
            }
        };
    }

    fn relu(&self) -> Tensor {
        let (m, n) = self.size;
        let mut data = vec![vec![0.0; n]; m];
        (0..m).for_each(|i| {
            (0..n).for_each(|j| {
                data[i][j] = match self.data[i][j] {
                    x if x >= 0.0 => x,
                    _ => 0.0,
                }
            })
        });

        Tensor {
            name: unary_label("ReLU".to_string(), self),
            data: data.clone(),
            size: (m, n),
            gradient: Gradient {
                last: Some(Tensor::from_vector(data)),
                operation: GradientOperation::ReLU(self.clone()),
                value: Some(Tensor::fill(m, n, 0.0)),
            }
            .wrap(),
        }
    }

    fn mean(&self) -> Tensor {
        let (m, n) = self.size;
        let mut sum = 0.0;
        (0..m).for_each(|i| (0..n).for_each(|j| sum += self.data[i][j]));
        let data = vec![vec![sum / (self.num_elements() as f64)]];

        Tensor {
            name: unary_label("Mean".to_string(), self),
            data: data.clone(),
            size: (m, n),
            gradient: Gradient {
                last: Some(Tensor::from_vector(data)),
                operation: GradientOperation::Mean(self.clone()),
                value: Some(Tensor::fill(m, n, 0.0)),
            }
            .wrap(),
        }
    }

    fn pow(&self, exp: i32) -> Tensor {
        let (m, n) = self.size;
        let mut data = vec![vec![0.0; n]; m];
        (0..m).for_each(|i| {
            (0..n).for_each(|j| {
                data[i][j] = self.data[i][j].powi(exp);
            })
        });

        Tensor {
            name: format!("({}^{})", format_name(self), exp),
            data: data.clone(),
            size: (m, n),
            gradient: Gradient {
                last: Some(Tensor::from_vector(data)),
                operation: GradientOperation::Pow(self.clone(), exp),
                value: Some(Tensor::fill(m, n, 0.0)),
            }
            .wrap(),
        }
    }
}

// Unary operations
impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<'a> Neg for &'a Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        let (m, n) = self.size;
        let mut data = vec![vec![0.0; n]; m];

        for i in 0..m {
            for j in 0..n {
                data[i][j] = -self[i][j];
            }
        }

        Tensor {
            name: unary_label("-".to_string(), self),
            data: data.clone(),
            size: (m, n),
            gradient: Gradient {
                last: Some(Tensor::from_vector(data)),
                operation: GradientOperation::Neg(self.clone()),
                value: Some(Tensor::fill(m, n, 0.0)),
            }
            .wrap(),
        }
    }
}

// In-place non-gradient operations
impl<'a> AddAssign<Tensor> for &'a mut Tensor {
    // NON-GRADIENT
    fn add_assign(&mut self, right: Tensor) {
        let (m, n) = self.size;
        assert!((m, n) == right.size);
        for i in 0..m {
            for j in 0..n {
                self[i][j] = self[i][j] + right[i][j];
            }
        }
    }
}

impl<'a> SubAssign<Tensor> for &'a mut Tensor {
    // NON-GRADIENT
    fn sub_assign(&mut self, right: Tensor) {
        let (m, n) = self.size;
        assert!((m, n) == right.size);
        for i in 0..m {
            for j in 0..n {
                self[i][j] = self[i][j] - right[i][j];
            }
        }
    }
}

// Binary operations
impl<'a> Add<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn add(self, right: &'a Tensor) -> Tensor {
        let (m, n) = self.size;
        let (m_2, n_2) = right.size;
        assert!((m, n) == right.size, "({}, {}) != ({}, {})", m, n, m_2, n_2);
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = self[i][j] + right[i][j];
            }
        }

        Tensor {
            name: binary_label(self, "+".to_string(), right),
            data: data.clone(),
            size: (m, n),
            gradient: Gradient {
                operation: GradientOperation::Add(self.clone(), right.clone()),
                last: Some(Tensor::from_vector(data)),
                value: Some(Tensor::fill(m, n, 0.0)),
            }
            .wrap(),
        }
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        &self + &rhs
    }
}

impl<'a> Sub<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn sub(self, right: &'a Tensor) -> Tensor {
        let (m, n) = self.size;
        let (m_2, n_2) = right.size;
        if n != n_2 {
            panic!("Incompatible dimensions: [{m}x{n}] - [{m_2}x{n_2}]");
        }
        assert!((m, n) == right.size);
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = self[i][j] - right[i][j];
            }
        }

        Tensor {
            name: binary_label(self, "-".to_string(), right),
            data: data.clone(),
            size: (m, n),
            gradient: Gradient {
                operation: GradientOperation::Sub(self.clone(), right.clone()),
                last: Some(Tensor::from_vector(data)),
                value: Some(Tensor::fill(m, n, 0.0)),
            }
            .wrap(),
        }
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, right: Tensor) -> Self::Output {
        &self - &right
    }
}

impl<'a> Mul<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, right: &'a Tensor) -> Tensor {
        let (m, n_1) = self.size;
        let (n_2, p) = right.size;

        // [m x n_1][n_2 x p] => [m x p]
        if n_1 != n_2 {
            panic!("Incompatible dimensions: [{m} x n_1: {n_1}][n_2: {n_2} x {p}], n_1 != n_2")
        }
        let mut data = vec![vec![0.0; p]; m];

        // TODO: loop over (i,j,k) tuples
        for i in 0..m {
            for j in 0..p {
                for k in 0..n_1 {
                    data[i][j] += self[i][k] * right[k][j];
                }
            }
        }

        Tensor {
            name: binary_label(self, "*".to_string(), right),
            data: data.clone(),
            size: (m, p),
            gradient: Gradient {
                operation: GradientOperation::Mul(self.clone(), right.clone()),
                last: Some(Tensor::from_vector(data)),
                value: Some(Tensor::fill(m, p, 0.0)),
            }
            .wrap(),
        }
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, right: Tensor) -> Self::Output {
        &self * &right
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, right: f64) -> Self::Output {
        &self * right
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, right: f64) -> Self::Output {
        let (m, n) = self.size;
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] += self[i][j] * right;
            }
        }

        Tensor {
            data: data.clone(),
            size: (m, n),
            ..Tensor::default()
        }
    }
}

impl Mul<Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        rhs * self
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        let (m, n) = self.size;
        let (m_2, n_2) = other.size;
        if m != m_2 || n != n_2 {
            return false;
        }
        for i in 0..m {
            for j in 0..n {
                if self[i][j] != other[i][j] {
                    return false;
                }
            }
        }
        true
    }
}
