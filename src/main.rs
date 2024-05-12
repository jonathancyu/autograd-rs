mod tensor;
use tensor::Tensor;


fn main() {
    let test: &[f64] = &[1.0; 500];
    let test2: &[f64] = &[3.0; 500];
    let a = Tensor {value: test};
    let b = Tensor {value: test2};
    let val: f64 = a.dot(&b);
    println!("Hello, world! {:?}, {:?}", b, val);
}
