#[cfg(test)]

mod tensor_tests {
    use llm_rs::tensor::Tensor;

    #[test]
    fn from_vector_sets_size() {
        let a = Tensor::from_vector(vec![vec![1.0, 2.0]]);
        assert_eq!((1, 2), a.size);
    }

    #[test]
    fn fill_sets_size() {
        let a = Tensor::fill(1, 2, 0.0);
        assert_eq!((1, 2), a.size);
    }

    #[test]
    fn from_array_sets_size() {
        let a = Tensor::from_array(&[&[1.0, 2.0]]);
        assert_eq!((1, 2), a.size);
    }

    #[test]
    fn indexmut_changes_value() {
        let mut a = Tensor::from_array(&[&[1.0, 2.0], &[3.0, 4.0]]);
        a[0][0] = 5.0;

        assert_eq!(5.0, a[0][0]);
        assert_eq!((2, 2), a.size);
    }

    #[test]
    fn transpose_flips_data() {
        let a = Tensor::from_array(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let expected = Tensor::from_array(&[&[1.0, 4.0], &[2.0, 5.0], &[3.0, 6.0]]);

        let result = a.transpose();

        assert_eq!(expected, result);
        assert_eq!((3, 2), result.size);
    }

    #[test]
    fn index_returns_correct_value() {
        let a = Tensor::from_array(&[&[1.0, 2.0], &[3.0, 4.0]]);

        assert_eq!(1.0, a[0][0]);
        assert_eq!(2.0, a[0][1]);
        assert_eq!(3.0, a[1][0]);
        assert_eq!(4.0, a[1][1]);
    }

    #[test]
    fn neg_returns_negative_value() {
        let a = Tensor::fill(2, 2, 2.0);

        let expected = Tensor::fill(2, 2, -2.0);

        assert_eq!(expected, -a);
    }

    #[test]
    fn add_returns_elementwise_sum() {
        let a = &Tensor::fill(2, 2, 2.0);
        let b = &Tensor::fill(2, 2, 1.0);

        let expected = Tensor::fill(2, 2, 3.0);

        let result = a + b;

        assert_eq!(expected, result);
        assert_eq!(a.size, result.size);
    }

    #[test]
    fn sub_returns_elementwise_diff() {
        let a = &Tensor::fill(2, 2, 2.0);
        let b = &Tensor::fill(2, 2, 3.0);

        let expected = Tensor::fill(2, 2, -1.0);
        let result = a - b;

        assert_eq!(expected, result);
        assert_eq!(a.size, result.size);
    }

    #[test]
    fn simple_multiplication() {
        let a = Tensor::fill(2, 2, 2.0);
        let b = Tensor::fill(2, 3, 1.0);

        let expected = Tensor::fill(2, 3, 4.0);

        let result = a * b;

        assert_eq!(expected, result);
        assert_eq!((2, 3), result.size);
    }

    #[test]
    fn complex_multiplication() {
        let a = Tensor::from_array(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = Tensor::from_array(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let expected = Tensor::from_array(&[&[9.0, 12.0, 15.0], &[19.0, 26.0, 33.0]]);

        let result = a * b;
        assert_eq!(expected, result);
        assert_eq!((2, 3), result.size);
    }

    #[test]
    fn pow_gives_correct_value() {
        let a = Tensor::from_array(&[&[1.0, 2.0], &[3.0, 4.0]]);

        let expected = Tensor::from_array(&[&[1.0, 4.0], &[9.0, 16.0]]);

        assert_eq!(expected, a.pow(2_i32));
    }

    #[test]
    fn apply_applies_function() {
        let a = Tensor::from_array(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let expected = Tensor::from_array(&[&[2.0, 3.0], &[4.0, 5.0]]);

        let result = a.apply(|i, j, data| data[i][j] + 1.0);
        println!("{:}\n{:}", a, result);

        assert_eq!(expected, result);
    }
}
