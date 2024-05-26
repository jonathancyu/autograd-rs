use core::f64;
use std::cell::RefCell;
use std::ops::{Index, IndexMut, Mul};
use std::fmt::{Debug, Display};
use std::rc::Rc;


pub struct Tensor {
    data: Vec<Vec<f64>>,
    pub parents: Vec<Rc<RefCell<Tensor>>>,
    pub grad: Rc<RefCell<f64>>, // TODO: make grad off by default
    pub backward: Box<dyn Fn(Vec<Tensor>, f64)>
}

//// Operations with Gradient
//impl<'a, 'b> Add<&'b Tensor<'b>> for &'a Tensor<'a> {
//    type Output = Tensor<'a>;
//    fn add(self, rhs: &'b Tensor<'b>) -> Self::Output {
//        let (m, n) = self.size();
//        assert!((m, n) == rhs.size());
//        let mut result = Tensor::fill(m, n, 0.0);
//        for i in 0..m {
//            for j in 0..n {
//                result[i][j] = self[i][j] + rhs[i][j];
//            }
//        }
//        result.set_parents(&vec![*self, *rhs]);
//        result.backward = Box::new(|parents, grad| {
//            assert!(parents.len() == 2);
//            *parents[0].grad.borrow_mut() += grad;
//        });
//
//        result
//    }
//}

//impl<'a> Neg for Tensor<'a> {
//    type Output = &'a Tensor<'a>;
//    fn neg(self) -> &'a Tensor<'a>  {
//        let (m, n) = self.size();
//        let result = & mut Tensor::fill(m, n, 0.0);
//
//        for i in 0..m {
//            for j in 0..n {
//                result[i][j] = -self[i][j];
//            }
//        }
//        result.set_parents(&vec![self]);
//        & result
//    }
//}


//impl<'a, 'b> Sub<&'b Tensor<'b>> for &'a Tensor<'a> {
//    type Output = Tensor<'a>;
//    fn sub(self, rhs: &'b Tensor<'b>) -> Self::Output {
//        self + (-rhs)
//    }
//}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, right: Tensor) -> Tensor {
        let (m, n) = self.size();
        let (n_2, p) = right.size();
 
        // [m x n][n x p] => [m x p]
        if n != n_2 {
            panic!("Incompatible dimensions")
        }
        let mut data = vec![vec![0.0; n]; m];

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
            parents: vec![Rc::new(RefCell::new(self)), Rc::new(RefCell::new(right))], // TODO:
            // cleaner way?
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


fn nop(_: Vec<Tensor>, _: f64) {}

impl Default for Tensor {
    fn default() -> Tensor {
        Tensor {
            data: vec![vec![0.0]],
            grad: Rc::new(RefCell::new(0.0)),
            parents: vec![],
            backward: Box::new(nop)
        }
    }
}

#[allow(dead_code)]
impl Tensor {
    // Constructors
    // TODO:
    //pub fn of<T, U>(&data: T) -> Tensor
    //where
    //    T: Iterator + Index<usize, Output = U>,
    //    U: Iterator + Index<usize, Output = f64>
    //{
    //    let (m, n) = (data.count(), data);
    //
    //    Tensor::singleton(0)
    //}
    
    pub fn from_vector(data: Vec<Vec<f64>>) -> Tensor {
        Tensor {
            data,
            ..Tensor::default()
        }
    }

    pub fn from_array(array: &[&[f64]]) -> Tensor {
        Tensor {
            data: array.iter()
                .map(|&row|
                    row.to_vec()
                ).collect::<Vec<_>>(),
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
            _ => panic!("Cannot call item() on a tensor with non-unit size")
        }
    }

    pub fn set_parents(mut self, parents: Vec<Rc<RefCell<Tensor>>>) {
        self.parents = parents.clone() // Copy the references to the parent tensor
    }

    // Operations
    //pub fn concat(&self, x: &Tensor) -> Tensor {
    //    let (m, _) = self.size();
    //    let (x_m, _) = x.size();
    //    if x_m != m {
    //        panic!("Expected {} got {}", m, x_m);
    //    }
    //
    //    let mut result = self.clone(); // TODO: is data the same pointer?
    //    for row in x.data.iter() {
    //        result.data.push(row.to_vec());
    //    }
    //    result
    //}

    //pub fn pow(self, rhs: i32) -> Tensor<'a> {
    //    let (m, n) = self.size();
    //    let mut result = Tensor::fill(m, n, 0.0);
    //    for i in 0..m {
    //        for j in 0..n {
    //            result[i][j] = self[i][j].powi(rhs);
    //        }
    //    }
    //
    //    result.set_parents(&vec![self, Tensor::singleton(rhs as f64)]);
    //
    //    //result.backward = Box::new(|grad| 
    //    //        self.grad = (rhs as f64) * (result.pow(rhs - 1)) * grad
    //    //    );
    //    result
    //}

    //pub fn sigmoid(self) -> Tensor<'a> {
    //    let e = 1.0_f64.exp();
    //
    //    let mut result = self.clone();
    //    let (m, n) = self.size();
    //    for i in 0..m {
    //        for j in 0..n {
    //            let y = self[i][j];
    //            result[i][j] = 1.0 / ( 1.0 + e.powf(y));
    //        }
    //    }
    //
    //    result
    //}

    pub fn size(&self) -> (usize, usize) {
        match self.data.is_empty() {
            true => (0, 0),
            false => match self.data[0].is_empty() {
                true => (0, 0),
                false => (self.data.len(), self.data[0].len())
            }
        }
    }
    
    pub fn transpose(&self) -> Tensor {
        let (m, n) = self.size();
        let mut data = vec![vec![0.0; m]; n];

        // TODO: how to do this with apply?
        for i in 0..n {
            for j in 0..m {
                data[i][j] = self[j][i];
            }
        }


        Tensor::from_vector(data)
    }
    pub fn apply(&self, fun: fn(usize, usize, &Tensor) -> f64) -> Tensor {
        // TODO: make this work for N-dimensional tensors
        let (m, n) = self.size();
        let mut data = vec![vec![0.0; n]; m];

        for i in 0..m {
            for j in 0..n {
                data[i][j] = fun(i, j, self);
            }
        }

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
    fn index_mut(& mut self, index: usize) -> & mut Vec<f64> {
        let (m, n) = self.size();
        assert!(index < m, "Index out of bounds");
        &mut self.data[index]
    }
}

impl<'a> Index<usize> for Tensor {
    type Output = Vec<f64>;
    fn index(&self, index: usize) -> &Vec<f64> {
        let (m, n) = self.size();
        assert!(index < m, "Index out of bounds");
        &self.data[index]
    }
}


impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = self.data.iter()
            .map(|row| {
                row.iter().map(|&x| {x.to_string()})
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
        let result = self.data.iter()
            .map(|row| {
                row.iter().map(|&x| {x.to_string()})
                    .collect::<Vec<String>>()
                .join(" ")
            })
            .collect::<Vec<String>>()
            .join("\n");
            

        writeln!(f, "{}", result)?;
        Ok(())
    }
}
