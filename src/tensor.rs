#[derive(Debug)]
pub struct Tensor<'a> {
    pub value: &'a [f64]
}

impl<'a> Tensor<'a> {
    pub fn dot(self, other: &Tensor) -> f64 {
        let other_value = other.value;
        let other_len = other_value.len();
        if self.value.len() != other_len {
            panic!("Tried to multiply {} by {}.", self.value.len(), other_len)
        }

        let mut sum = 0.0;
        for i in 0..other_len {
            sum += self.value[i] * other_value[i];
        }

        println!("Sum: {}", sum);

        sum
    }

}
