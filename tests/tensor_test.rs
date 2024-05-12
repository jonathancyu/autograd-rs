#[cfg(test)]
mod tensor_tests {
    use llm_rs::tensor::Tensor;

    #[test]
    fn simple_multiplication() {
        let a = Tensor::fill(2, 2, 2.0);
        let b = Tensor::fill(2, 3, 1.0);

        let expected = Tensor::fill(2, 3, 4.0);

        assert_eq!(expected, a * b);
    }

    #[test]
    fn complex_multiplication() {
        let a = Tensor::of(&[
            &[1.0, 2.0],
            &[3.0, 4.0]
        ]);
        let b = Tensor::of(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
        ]);

        let expected = Tensor::of(&[
            &[ 9.0, 12.0, 15.0],
            &[19.0, 26.0, 33.0]
        ]);

        assert_eq!(expected, a * b);
    }

}
