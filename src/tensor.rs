use core::{f64, panic};
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::ops::{Index, IndexMut};
use std::rc::Rc;

use crate::operations::{GradientNode, GradientOperation};


pub struct TensorData {
    pub operation: GradientOperation,
    pub last: Option<Tensor>,
    pub name: String,
    pub grad: Option<Tensor>, // Shouldn't grad be ties to operation?
}

impl Default for TensorData {
    fn default() -> Self {
        TensorData {
            name: String::new(),
            operation: GradientOperation::None,
            last: None,
            grad: None
        }
    }
}

impl TensorData {
    pub fn wrap(self) -> Rc<RefCell<TensorData>> {
        Rc::new(RefCell::new(self))
    }
}


pub struct Tensor {
    pub data: Vec<Vec<f64>>,
    pub metadata: Rc<RefCell<TensorData>>,
}

// Housekeeping
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
    pub fn backward(&self) {
        let binding = self.metadata();
        let metadata = binding.borrow_mut();
        metadata.backward();
    }

    pub fn grad(&self) -> Tensor {
        let binding = self.metadata();
        let metadata = binding.borrow_mut();
        metadata.grad().clone()
    }

    pub fn set_grad(&self, grad: Tensor) {
        let binding = self.metadata();
        let mut metadata = binding.borrow_mut();
        metadata.grad = Some(grad);
    }

    pub fn with_grad(self) -> Self {
        self.set_grad(Tensor::empty());
        self
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

    pub fn metadata(&self) -> Rc<RefCell<TensorData>> {
        self.metadata.clone()
    }

    pub fn from_vector(data: Vec<Vec<f64>>) -> Tensor {
        Tensor {
            data: data.clone(),
            metadata: TensorData::default().wrap(),
        }
    }

    pub fn from_array(array: &[&[f64]]) -> Tensor {
        Tensor::from_vector(
            array.iter().map(|&row| {
                row.to_vec()
            }).collect::<Vec<_>>(),
        )
    }

    pub fn empty() -> Tensor {
        let data = vec![vec![]];
        Tensor {
            data: data.clone(),
            metadata: TensorData::default().wrap()
        }
    }

    pub fn singleton(value: f64) -> Tensor {
        Tensor::fill(1, 1, value)
    }

    pub fn fill(m: usize, n: usize, value: f64) -> Tensor {
        let data = vec![vec![value; n]; m];
        Tensor {
            data: data.clone(),
            metadata: TensorData::default().wrap()
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
        let data = &self.data;
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
        let result = self.data
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
        let result = self.data
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
