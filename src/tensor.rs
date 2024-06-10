use core::f64;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};
use std::rc::Rc;

pub struct Tensor {
    data: Vec<Vec<f64>>,
    pub grad: Rc<RefCell<f64>>, // TODO: make grad off by default, TODO: should this be a Tensor?
    grad_fn: Box<dyn Fn(f64)>,
}

pub trait Backward {
    fn backward(self, grad: f64);
}

impl Backward for Tensor {
    fn backward(self, grad: f64) {
        (self.grad_fn)(grad)
    }
}

// Operations with Gradient
impl<'a> Add<Tensor> for &'a Tensor {
    type Output = Tensor;
    fn add(self, right: Tensor) -> Self::Output {
        let (m, n) = self.size();
        assert!((m, n) == right.size());
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = self[i][j] + right[i][j];
            }
        }

        let left = self.grad.clone();
        let right = right.grad.clone();

        Tensor {
            data,
            grad_fn: Box::new(move |grad: f64| {
                *left.borrow_mut() += grad;
                *right.borrow_mut() += grad;
            }),
            ..Tensor::default()
        }
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
            data,
            ..Tensor::default()
        }
    }
}

impl<'a> Sub<Tensor> for &'a Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        self + -(&rhs)
    }
}

impl<'a> Mul<Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, right: Tensor) -> Tensor {
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
            data,
            ..Tensor::default()
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
                if self[i][j] != other.data[i][j] {
                    return false;
                }
            }
        }
        true
    }
}

fn nop(_grad: f64) {}

impl Default for Tensor {
    fn default() -> Tensor {
        Tensor {
            data: vec![vec![0.0]],
            grad: Rc::new(RefCell::new(0.0)),
            grad_fn: Box::new(nop),
        }
    }
}

#[allow(dead_code)]
impl Tensor {
    pub fn from_vector(data: Vec<Vec<f64>>) -> Tensor {
        Tensor {
            data,
            ..Tensor::default()
        }
    }

    pub fn from_array(array: &[&[f64]]) -> Tensor {
        Tensor {
            data: array.iter().map(|&row| row.to_vec()).collect::<Vec<_>>(),
            ..Tensor::default()
        }
    }

    pub fn singleton(value: f64) -> Tensor {
        Tensor::fill(1, 1, value)
    }

    pub fn fill(m: usize, n: usize, value: f64) -> Tensor {
        Tensor {
            data: vec![vec![value; n]; m],
            ..Tensor::default()
        }
    }

    pub fn zeros(m: usize, n: usize) -> Tensor {
        Tensor::fill(m, n, 0.0)
    }
    pub fn ones(m: usize, n: usize) -> Tensor {
        Tensor::fill(m, n, 1.0)
    }

    pub fn item(&self) -> f64 {
        match self.size() {
            (1, 1) => self[0][0],
            _ => panic!("Cannot call item() on a tensor with non-unit size"),
        }
    }

    // Operations
    pub fn concat(&self, x: &Tensor) -> Tensor {
        let (m, _) = self.size();
        let (x_m, _) = x.size();
        if x_m != m {
            panic!("Expected {} got {}", m, x_m);
        }

        let mut result = self.clone(); // TODO: is data the same pointer?
        for row in x.data.iter() {
            result.data.push(row.to_vec());
        }
        result
    }

    pub fn pow(self, rhs: i32) -> Tensor {
        let (m, n) = self.size();
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = self[i][j].powi(rhs);
            }
        }

        Tensor {
            data,
            ..Tensor::default()
        }
    }

    pub fn size(&self) -> (usize, usize) {
        match self.data.is_empty() {
            true => (0, 0),
            false => match self.data[0].is_empty() {
                true => (0, 0),
                false => (self.data.len(), self.data[0].len()),
            },
        }
    }

    pub fn transpose(&self) -> Tensor {
        let (m, n) = self.size();
        let mut data = vec![vec![0.0; m]; n];

        // TODO: how to do this with apply?
        (0..n).for_each(|i| {
            (0..m).for_each(|j| {
                data[i][j] = self[j][i];
            });
        });

        Tensor::from_vector(data)
    }

    pub fn apply(&self, fun: fn(usize, usize, &Tensor) -> f64) -> Tensor {
        // TODO: make this work for N-dimensional tensors
        let (m, n) = self.size();
        let mut data = vec![vec![0.0; n]; m];

        (0..m).for_each(|i| {
            (0..n).for_each(|j| {
                data[i][j] = fun(i, j, self);
            });
        });

        Tensor::from_vector(data)
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let (m, n) = self.size();
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = self[i][j];
            }
        }
        Tensor {
            data,
            ..Tensor::default()
        }
    }
}

impl IndexMut<usize> for Tensor {
    //type Output = &'a mut Vec<f64>;
    fn index_mut(&mut self, index: usize) -> &mut Vec<f64> {
        let (m, _n) = self.size();
        assert!(index < m, "Index out of bounds");
        &mut self.data[index]
    }
}

impl Index<usize> for Tensor {
    type Output = Vec<f64>;
    fn index(&self, index: usize) -> &Vec<f64> {
        let (m, _n) = self.size();
        assert!(index < m, "Index out of bounds");
        &self.data[index]
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = self
            .data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(" ")
            })
            .collect::<Vec<String>>()
            .join("\n");

        writeln!(f, "{}", result)?;
        Ok(())
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = self
            .data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(" ")
            })
            .collect::<Vec<String>>()
            .join("\n");

        writeln!(f, "{}", result)?;
        Ok(())
    }
}
