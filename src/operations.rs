use std::{
    cell::RefCell,
    ops::{ Add, AddAssign, Mul, Neg, Sub, SubAssign },
    rc::Rc,
};

use crate::tensor::Tensor;

pub struct Gradient {
    pub operation: GradientOperation,
    pub last: Option<Tensor>,
    pub value: Option<Tensor>, // Shouldn't grad be ties to operation?
}

impl Default for Gradient {
    fn default() -> Self {
        Gradient {
            operation: GradientOperation::None,
            last: None,
            value: None
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
    Unary(Rc<RefCell<Gradient>>)
}

// TODO: no words
#[derive(Clone)]
pub enum GradientOperation {
    None,
    Neg(Tensor),
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor)
}

pub trait Differentiable {
    fn grad(&self) -> Tensor;
    fn add_grad(&self, grad: Tensor);
    fn last(&self) -> Tensor;

    fn backward(&self);
}

impl Differentiable for Tensor {
    fn grad(&self) -> Tensor {
        let gradient = self.gradient.borrow();
        let value = gradient.value.clone().expect("Tensor doesn't have grad enabled");
        value.clone()
    }

    fn add_grad(&self, grad: Tensor) {
        let mut gradient = self.gradient.borrow_mut();
        let value = gradient.value.clone().expect("Tensor doesn't have grad enabled");
        gradient.value = Some(value + grad);
    }

    fn last(&self) -> Tensor {
        let gradient = self.gradient.borrow();
        gradient.last.clone().expect("Tensor doesn't have last value")
    }

    fn backward(&self) {
        let grad = self.grad().clone();
        let gradient = self.gradient.borrow_mut();
        match &gradient.operation {
            GradientOperation::None => {},
            GradientOperation::Neg(a) => {
                // y = -a
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * -1
                a.add_grad(-grad);
                a.backward();
            },
            GradientOperation::Add(a, b) => {
                // y = a + b
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * 1
                // b.grad = dL/db = (dL/dy)(dy/db) = grad * 1
                a.add_grad(-grad.clone());
                a.add_grad(-grad.clone());
                a.backward();
                b.backward();
            },
            GradientOperation::Sub(a, b) => {
                // y = a - b
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * 1
                // b.grad = dL/db = (dL/dy)(dy/db) = grad * -1
                a.add_grad(grad.clone());
                b.add_grad(-grad.clone());
                a.backward();
                b.backward();
            },
            GradientOperation::Mul(a, b) => {
                // y = a * b
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * b
                // b.grad = dL/db = (dL/dy)(dy/db) = grad * a
                let a_last = a.last();
                let b_last = b.last();
                a.add_grad(grad.clone() * b_last);
                a.add_grad(grad.clone() * a_last);
                a.backward();
                b.backward();
            },
        };
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
        let (m, n) = self.size();
        let mut data = vec![vec![0.0; n]; m];

        for i in 0..m {
            for j in 0..n {
                data[i][j] = -self[i][j];
            }
        }

        Tensor {
            name: self.name.clone(),
            data: data.clone(),
            gradient: Gradient {
                last: Some(self.clone()),
                operation: GradientOperation::Neg(self.clone()),
                value: Some(Tensor::fill(m, n, 0.0)),
            }.wrap(),
        }
    }
}
// In-place non-gradient operations
impl<'a> AddAssign<Tensor> for &'a mut Tensor {
    // NON-GRADIENT
    fn add_assign(&mut self, right: Tensor) {
        let (m, n) = self.size();
        assert!((m, n) == right.size());
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
        let (m, n) = self.size();
        assert!((m, n) == right.size());
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
        let (m, n) = self.size();
        assert!((m, n) == right.size());
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = self[i][j] + right[i][j];
            }
        }


        Tensor {
            name: format!("({} + {})", self.name, right.name),
            data: data.clone(),
            gradient: Gradient {
                operation: GradientOperation::Add(self.clone(), right.clone()),
                last: Some(self.clone()),
                value: Some(Tensor::fill(m, n, 0.0)),
            }.wrap(),
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
        let (m, n) = self.size();
        assert!((m, n) == right.size());
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = self[i][j] - right[i][j];
            }
        }


        Tensor {
            name: format!("({} - {})", self.name, right.name),
            data: data.clone(),
            gradient: Gradient {
                operation: GradientOperation::Add(self.clone(), right.clone()),
                last: Some(self.clone()),
                value: Some(Tensor::fill(m, n, 0.0)),
            }.wrap(),
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
        let (m, n) = self.size();
        let (n_2, p) = right.size();

        // [m x n][n x p] => [m x p]
        if n != n_2 {
            panic!("Incompatible dimensions")
        }
        let mut data = vec![vec![0.0; p]; m];

        // TODO: loop over (i,j,k) tuples
        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    data[i][j] += self[i][k] * right[k][j];
                }
            }
        }

        Tensor {
            name: format!("({} * {})", self.name, right.name),
            data: data.clone(),
            gradient: Gradient {
                last: Some(self.clone()),
                value: Some(Tensor::fill(m, p, 0.0)),
                ..Gradient::default()
            }.wrap(),
        }

    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, right: Tensor) -> Self::Output {
        &self * &right
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        let (m, n) = self.size();
        let (m_2, n_2) = other.size();
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

