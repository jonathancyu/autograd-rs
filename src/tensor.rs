use core::f64;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};
use std::rc::Rc;

pub struct TensorData {
    pub last: Vec<Vec<f64>>,
    pub name: String,
    pub grad: f64, // TODO: make grad off by default, TODO: should this be a Tensor?
    pub grad_fn: Box<dyn Fn(f64)>,
}
impl TensorData {
    fn wrap(self) -> Rc<RefCell<TensorData>> {
        Rc::new(RefCell::new(self))
    }

    fn default() -> TensorData {
        TensorData {
            last: vec![vec![0.0]],
            name: String::new(),
            grad: 0.0,
            grad_fn: Box::new(nop),
        }
    }
}

pub struct Tensor {
    data: Vec<Vec<f64>>,
    metadata: Rc<RefCell<TensorData>>,
}

pub trait Backward {
    fn backward(self);
}

impl Backward for &Tensor {
    fn backward(self) {
        let binding = self.metadata();
        let metadata = binding.borrow();
        println!("{}: {}", metadata.name, metadata.grad);

        (metadata.grad_fn)(metadata.grad)
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
                last: data.clone(),
                name,
                grad_fn: Box::new(move |grad: f64| {
                    // y = -a
                    // dL/da = (dL/dy)(dy/da) = grad * -1
                    metadata.borrow_mut().grad -= grad;
                }),
                ..TensorData::default()
            }.wrap(),
        }
    }
}

// Binary operations
impl<'a> Add<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn add(self, right: &'a Tensor) -> Self::Output {
        let (m, n) = self.size();
        assert!((m, n) == right.size());
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = self[i][j] + right[i][j];
            }
        }

        let name = format!("({} + {})", self.name(), right.name());

        let left = self.metadata.clone();
        let right = right.metadata.clone();
        Tensor {
            data: data.clone(),
            metadata: TensorData {
                name,
                last: data.clone(),
                grad_fn: Box::new(move |grad: f64| {
                    // y = a + b
                    // dL/da = (dL/dy)(dy/da) = grad * 1
                    // dL/db = (dL/dy)(dy/db) = grad * 1
                    let mut left = left.borrow_mut();
                    let mut right = right.borrow_mut();
                    left.grad += grad;
                    right.grad += grad;

                    (left.grad_fn)(grad);
                    (right.grad_fn)(grad)
                }),
                ..TensorData::default()
            }.wrap(),
        }
    }
}


impl<'a> Sub<&'a Tensor> for &'a Tensor {
    type Output = Tensor;
    fn sub(self, right: &'a Tensor) -> Self::Output {
        let name = format!("({} - {})", self.name(), right.name());
        (self + &(-right)).named(name) // TODO: is this correct?
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
        let name = format!("({} * {})", self.name(), right.name());

        let left = self.metadata.clone();
        let right = right.metadata.clone();

        Tensor {
            data: data.clone(),
            metadata: TensorData {
                name,
                last: data.clone(),
                grad_fn: Box::new(move |grad: f64| {
                    // y = a * b
                    // dL/da = (dL/dy)(dy/da) = grad * b
                    // dL/db = (dL/dy)(dy/db) = grad * a
                    left.borrow_mut().grad += grad;
                    right.borrow_mut().grad += grad;

                    let mut left = left.borrow_mut();
                    let mut right = right.borrow_mut();
                    left.grad += grad;
                    right.grad += grad;

                    (left.grad_fn)(grad);
                    (right.grad_fn)(grad)
                }),
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

// Housekeeping
fn nop(_grad: f64) {}

impl Default for Tensor {
    fn default() -> Tensor {
        Tensor {
            data: vec![vec![0.0]],
            metadata: TensorData::default().wrap(),
        }
    }
}

// TODO: clean up this dumping ground
#[allow(dead_code)]
impl Tensor {
    pub fn set_grad(&self, grad: f64) {
        // TODO: there must be a better way.. there must be
        let binding = self.metadata();
        let mut metadata = binding.borrow_mut();
        metadata.grad = grad;
    }

    pub fn named(self, name: String) -> Self {
        let binding = self.metadata();
        let mut metadata = binding.borrow_mut();
        metadata.name = name;
        self
    }

    pub fn name(&self) -> String {
        let binding = self.metadata();
        let metadata = binding.borrow_mut();
        metadata.name.clone()
    }

    pub fn data(&self) -> Vec<Vec<f64>> {
        let binding = self.metadata();
        let metadata = binding.borrow_mut();
        metadata.last.clone()
    }

    pub fn metadata(&self) -> Rc<RefCell<TensorData>> {
        self.metadata.clone()
    }

    pub fn from_vector(data: Vec<Vec<f64>>) -> Tensor {
        Tensor {
            data: data.clone(),
            metadata: TensorData {
                last: data.clone(),
                ..TensorData::default()
            }.wrap()
        }
    }

    pub fn from_array(array: &[&[f64]]) -> Tensor {
        Tensor::from_vector(
            array.iter().map(|&row| {
                row.to_vec()
            }).collect::<Vec<_>>(),
        )
    }

    pub fn singleton(value: f64) -> Tensor {
        Tensor::fill(1, 1, value)
    }

    pub fn fill(m: usize, n: usize, value: f64) -> Tensor {
        let data = vec![vec![value; n]; m];
        Tensor {
            data: data.clone(),
            metadata: TensorData {
                last: data.clone(),
                ..TensorData::default()
            }.wrap()
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

    pub fn grad(&self) -> f64 {
        let binding = self.metadata();
        let metadata = binding.borrow();
        metadata.grad
    }

    // Operations
    // pub fn concat(&self, x: &Tensor) -> Tensor {
    //     let (m, _) = self.size();
    //     let (x_m, _) = x.size();
    //     if x_m != m {
    //         panic!("Expected {} got {}", m, x_m);
    //     }
    //
    //     let mut result = self.clone(); // TODO: is data the same pointer?
    //     for row in x.data.iter() {
    //         result.data.push(row.to_vec());
    //     }
    //     result
    // }

    pub fn pow(&self, rhs: i32) -> Tensor {
        let (m, n) = self.size();
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = self[i][j].powi(rhs);
            }
        }

        Tensor::from_vector(data)
    }

    pub fn size(&self) -> (usize, usize) {
        let data = self.data();
        match data.is_empty() {
            true => (0, 0),
            false => match data[0].is_empty() {
                true => (0, 0),
                false => (data.len(), data[0].len()),
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
        Tensor::from_vector(data)
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
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
            .data()
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
            .data()
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
trait ToGraphviz {
    fn to_dot() -> String;
}

impl ToGraphviz for Tensor {
    fn to_dot() -> String {
        "".to_string()
    }
}
