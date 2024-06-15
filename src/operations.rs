use std::{
    cell::RefCell,
    ops::{ Add, AddAssign, Mul, Neg, Sub, SubAssign },
    rc::Rc,
};

use crate::tensor::{Tensor, TensorData};

#[derive(Clone)]
pub enum Parents {
    None,
    Unary(Rc<RefCell<TensorData>>)
}

// TODO: no words
#[derive(Clone)]
pub enum GradientOperation {
    None,
    Neg(Rc<RefCell<TensorData>>),
    Add(Rc<RefCell<TensorData>>, Rc<RefCell<TensorData>>),
    Sub(Rc<RefCell<TensorData>>, Rc<RefCell<TensorData>>),
    Mul(Rc<RefCell<TensorData>>, Rc<RefCell<TensorData>>)
}
pub trait GradientNode {
    fn backward(&self);
    fn grad(&self) -> Tensor;
}

impl GradientNode for TensorData {
    fn grad(&self) -> Tensor {
        self.grad.clone().unwrap_or_else(|| {
            panic!("Gradient is not enabled on '{}'", self.name)
        })
    }

    fn backward(&self) {
        match &self.operation {
            GradientOperation::None => {},
            GradientOperation::Neg(a) => {
                // y = -a
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * -1
                let mut a = a.borrow_mut();
                a.grad = Some((&a.grad() + &-(&self.grad())).clone());
                a.backward();
            },
            GradientOperation::Add(a, b) => {
                // y = a + b
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * 1
                // b.grad = dL/db = (dL/dy)(dy/db) = grad * 1
                let mut a = a.borrow_mut();
                let mut b = b.borrow_mut();
                a.grad = Some(( &a.grad() + &self.grad() ).clone());
                b.grad = Some(( &b.grad() + &self.grad() ).clone());
                a.backward();
                b.backward();
            },
            GradientOperation::Sub(a, b) => {
                // y = a - b
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * 1
                // b.grad = dL/db = (dL/dy)(dy/db) = grad * -1
                let mut a = a.borrow_mut();
                let mut b = b.borrow_mut();
                a.grad = Some(( &a.grad() + &self.grad() ).clone());
                b.grad = Some(( &b.grad() + &-(&self.grad()) ).clone());
                a.backward();
                b.backward();

            },
            GradientOperation::Mul(a, b) => {
                // y = a * b
                // a.grad = dL/da = (dL/dy)(dy/da) = grad * b
                // b.grad = dL/db = (dL/dy)(dy/db) = grad * a
                let mut a = a.borrow_mut();
                let mut b = b.borrow_mut();
                let b_last = &b.last.clone().expect("last is None");
                let a_last = &a.last.clone().expect("last is None");
                a.grad = Some((&a.grad() + &(&self.grad() * b_last)).clone());
                b.grad = Some((&b.grad() + &(&self.grad() * a_last)).clone());

                a.backward();
                b.backward();
            },
        };
    }

}

// Unary operations
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

        let name = self.name();
        let metadata = self.metadata.clone();
        Tensor {
            data: data.clone(),
            metadata: TensorData {
                name,
                last: Some(self.clone()),
                operation: GradientOperation::Neg(metadata.clone()),
                ..TensorData::default()
            }.wrap(),
        }
    }
}
// In-place non-gradient operations
impl AddAssign<Tensor> for Tensor {
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

impl SubAssign<Tensor> for Tensor {
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


        let a: Rc<RefCell<TensorData>> = self.metadata.clone();
        let b: Rc<RefCell<TensorData>> = right.metadata.clone();
        Tensor {
            data: data.clone(),
            metadata: TensorData {
                name: format!("({} + {})", self.name(), right.name()),
                operation: GradientOperation::Add(a.clone(), b.clone()),
                last: Some(self.clone()),
                ..TensorData::default()
            }.wrap(),
        }
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


        let a: Rc<RefCell<TensorData>> = self.metadata.clone();
        let b: Rc<RefCell<TensorData>> = right.metadata.clone();
        Tensor {
            data: data.clone(),
            metadata: TensorData {
                name: format!("({} - {})", self.name(), right.name()),
                operation: GradientOperation::Add(a.clone(), b.clone()),
                last: Some(self.clone()),
                ..TensorData::default()
            }.wrap(),
        }
    }
}

impl<'a> Mul<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, right: &'a Tensor) ->  Tensor {
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
            data: data.clone(),
            metadata: TensorData {
                name: format!("({} * {})", self.name(), right.name()),
                last: Some(self.clone()),
                ..TensorData::default()
            }.wrap(),
        }

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

