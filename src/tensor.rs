use core::f64;
use std::ops::{Index, Mul, Add, Sub, Neg};
use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<Vec<f64>>,
    pub m: usize,
    pub n: usize,
}

impl Tensor {
    #[allow(dead_code)]
    pub fn of(array: &[&[f64]]) -> Tensor {
        let m = array.len();
        let n = array[0].len();

        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = array[i][j];
            }

        }
        Tensor { data, m, n }
    }

    pub fn singleton(value: f64) -> Tensor {
        Tensor {
            data: vec![vec![value]],
            m: 1,
            n: 1
        }
    }

    pub fn fill(m: usize, n: usize, c: f64) -> Tensor {
        let data = vec![vec![c; n]; m];
        Tensor { data, m, n }
    }

    pub fn zeros(m: usize, n: usize) -> Tensor {
        Tensor::fill(m, n, 0.0)
    }
    pub fn ones(m: usize, n: usize) -> Tensor {
        Tensor::fill(m, n, 1.0)
    }

    pub fn concat(&self, x: &Tensor) -> Tensor {
        if self.n != x.n || x.m != 1 {
            panic!("Expected {} got {}", self.n, x.n);
        }
        let mut result = self.clone(); // TODO: is data the same pointer?
        for row in x.data.iter() {
            result.data.push(row.to_vec());
            result.m += 1;
        }
        result
    }

    pub fn size(&self) -> (usize, usize) {
        (self.m, self.n)
    }
}

impl Index<usize> for Tensor {
    type Output = Vec<f64>;
    fn index(&self, index: usize) -> &Vec<f64> {
        assert!(index < self.m, "Index out of bounds");

        &self.data[index]
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        let (m, n) = self.size();
        let mut data = vec![vec![0.0; n]; m];
        
        for i in 0..m {
            for j in 0..n {
                data[i][j] = -self[i][j];
            }
        }
        Tensor { data, m, n }
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        let (m, n) = self.size();
        assert!((m, n) == rhs.size());
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = self[i][j] - rhs[i][j];
            }
        }

        Tensor { data, m, n }
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        let (m, n) = self.size();
        assert!((m, n) == rhs.size());
        let mut data = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                data[i][j] = self[i][j] + rhs[i][j];
            }
        }

        Tensor { data, m, n }
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, right: Tensor) -> Tensor {
        let (n, m) = (self.m, self.n);
        let (m_2, p) = (right.m, right.n);
 
        // [n x m][m x p] => [n x p]
        if m != m_2 {
            panic!("Incompatible dimensions")
        }
        let mut data = vec![vec![0.0; p]; n];
        //println!("{}, {}, {}, {:?}", n, m, p, data);
        for i in 0..n {
            for j in 0..p {
                for k in 0..m {
                    data[i][j] += self[i][k] * right[k][j];
                }
            }
        }
        Tensor { data, m: n, n: p }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if self.m != other.m || self.n != other.n {
            return false;
        }
        for i in 0..self.m {
            for j in 0..self.n {
                if self[i][j] != other.data[i][j] {
                    return false;
                }
            }
        }
        true
    }
}


// TODO: impl
impl Display for Tensor {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}
