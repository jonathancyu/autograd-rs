use std::{
    cell::RefCell,
    ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign},
    rc::Rc,
};

use crate::tensor::Tensor;

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
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
}

pub trait Differentiable {
    fn grad(&self) -> Tensor;
    fn with_grad(self) -> Self;
    fn set_grad(&self, grad: Tensor);
    fn add_grad(&self, grad: Tensor);

    fn last(&self) -> Tensor;

    fn backward(&self);

    fn relu(&self) -> Tensor;
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

    fn set_grad(&self, grad: Tensor) {
        let mut gradient = self.gradient.borrow_mut();
        gradient.value = Some(grad)
    }

    fn add_grad(&self, grad: Tensor) {
        let mut gradient = self.gradient.borrow_mut();
        let value = gradient
            .value
            .clone()
            .expect("Tensor doesn't have grad enabled");
        gradient.value = Some(value + grad);
    }

    fn last(&self) -> Tensor {
        let gradient = self.gradient.borrow();
        match &gradient.last {
            Some(value) => value.clone(),
            None => panic!("Tensor doesn't have last value"),
        }
    }

    fn backward(&self) {
        let grad = self.grad();
        let gradient = self.gradient.borrow();
        match &gradient.operation {
            GradientOperation::None => {}
            GradientOperation::Neg(a) => {
                // y = -a
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * -1
                a.add_grad(-grad.clone());
                a.backward();
            }
            GradientOperation::Add(a, b) => {
                // y = a + b
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * 1
                // b.grad = dL/db = (dL/dy)(dy/db) = grad * 1
                a.add_grad(grad.clone());
                a.backward();
                b.add_grad(grad.clone());
                b.backward();
            }
            GradientOperation::Sub(a, b) => {
                // y = a - b
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * 1
                // b.grad = dL/db = (dL/dy)(dy/db) = grad * -1
                a.add_grad(grad.clone());
                a.backward();
                b.add_grad(-grad.clone());
                b.backward();
            }
            GradientOperation::Mul(a, b) => {
                // y = a * b
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * b
                // b.grad = dL/db = (dL/dy)(dy/db) = grad * a
                let a_last = a.last();
                let b_last = b.last();
                a.add_grad(grad.clone() * b_last);
                a.backward();
                b.add_grad(grad.clone() * a_last);
                b.backward();
            }
            GradientOperation::ReLU(a) => {
                // y = [ x >= 0: x, x < 0: 0 ]
                // dy/dx = [x >= 0: 1, x < 0: 0]
                let a_last = a.last();
                a.add_grad(
                    a_last.apply(|i, j, last| if last[i][j] >= 0.0 { grad[i][j] } else { 0.0 }),
                )
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
        assert!((m, n) == right.size);
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
            panic!("Incompatible dimensions: [{m} x {n_1}][{n_2} x {p}], n_1 != n2")
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
