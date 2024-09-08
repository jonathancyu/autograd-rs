use core::{f64, panic};
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, Index, IndexMut, SubAssign};
use std::rc::Rc;

use crate::operations::Gradient;

pub struct Tensor {
    pub name: String,
    pub data: Vec<Vec<f64>>,
    pub size: (usize, usize),
    pub gradient: Rc<RefCell<Gradient>>,
}

// Housekeeping
impl Default for Tensor {
    fn default() -> Tensor {
        Tensor {
            name: String::new(),
            data: vec![vec![0.0]],
            size: (1, 1),
            gradient: Gradient::default().wrap(),
        }
    }
}

// TODO: clean up this dumping ground
#[allow(dead_code)]
impl Tensor {
    pub fn named(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn metadata(&self) -> Rc<RefCell<Gradient>> {
        self.gradient.clone()
    }

    pub fn from_vector(data: Vec<Vec<f64>>) -> Tensor {
        Tensor {
            data: data.clone(),
            size: Tensor::get_size(&data),
            ..Tensor::default()
        }
    }

    pub fn from_array(array: &[&[f64]]) -> Tensor {
        Tensor::from_vector(array.iter().map(|&row| row.to_vec()).collect::<Vec<_>>())
    }

    pub fn empty() -> Tensor {
        let data = vec![vec![]];
        Tensor {
            data: data.clone(),
            size: (1, 0), // TODO: ..?
            ..Tensor::default()
        }
    }

    pub fn singleton(value: f64) -> Tensor {
        Tensor::fill(1, 1, value)
    }

    pub fn fill(m: usize, n: usize, value: f64) -> Tensor {
        let data = vec![vec![value; n]; m];
        Tensor::from_vector(data)
    }

    pub fn zeros(m: usize, n: usize) -> Tensor {
        Tensor::fill(m, n, 0.0)
    }

    pub fn ones(m: usize, n: usize) -> Tensor {
        Tensor::fill(m, n, 1.0)
    }

    pub fn num_elements(&self) -> i32 {
        let (m, n) = self.size;
        (m as i32) * (n as i32)
    }

    pub fn item(&self) -> f64 {
        match self.size {
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

    fn get_size(data: &[Vec<f64>]) -> (usize, usize) {
        match data.is_empty() {
            true => (0, 0),
            false => match data[0].is_empty() {
                true => (0, 0),
                false => (data.len(), data[0].len()),
            },
        }
    }

    pub fn transpose(&self) -> Tensor {
        let (m, n) = self.size;
        let mut data = vec![vec![0.0; m]; n];

        // TODO: how to do this with apply?
        (0..n).for_each(|i| {
            (0..m).for_each(|j| {
                data[i][j] = self[j][i];
            });
        });

        Tensor::from_vector(data)
    }

    pub fn apply(&self, fun: impl Fn(usize, usize, &Tensor) -> f64) -> Tensor {
        // TODO: make this work for N-dimensional tensors
        let (m, n) = self.size;
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
        let (m, n) = self.size;
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = self[i][j];
            }
        }
        Tensor {
            data: self.data.clone(),
            size: (m, n),
            name: self.name.clone(),
            gradient: Rc::clone(&self.gradient),
        }
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let (m, _n) = self.size;
        assert!(index < m, "Index out of bounds");
        &mut self.data[index]
    }
}

impl Index<usize> for Tensor {
    type Output = Vec<f64>;
    fn index(&self, index: usize) -> &Vec<f64> {
        let (m, _n) = self.size;
        assert!(index < m, "Index out of bounds");
        &self.data[index]
    }
}

impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, right: &Tensor) {
        assert_eq!(self.size, right.size, "Sizes must be equal");
        let (m, n) = self.size;
        (0..m).for_each(|i| (0..n).for_each(|j| self.data[i][j] += right[i][j]));
    }
}

impl SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, right: &Tensor) {
        assert_eq!(self.size, right.size, "Sizes must be equal");
        let (m, n) = self.size;
        (0..m).for_each(|i| (0..n).for_each(|j| self.data[i][j] -= right[i][j]));
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self::Display::fmt(&self, f)
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = self
            .data
            .iter()
            .map(|row| {
                format!(
                    "[{}]",
                    row.iter()
                        .map(|&x| x.to_string())
                        .collect::<Vec<String>>()
                        .join(" ")
                )
            })
            .collect::<Vec<String>>()
            .join(" ");

        write!(f, "[{}]", result)?;
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
