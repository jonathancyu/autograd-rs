#[cfg(test)]
mod tensor_tests {
    use llm_rs::tensor::Tensor;

    #[test]
    fn indexmut_changes_value() {
        let mut a = Tensor::of(&[
            &[1.0, 2.0],
            &[3.0, 4.0]
        ]);
        a[0][0] = 5.0;

        assert_eq!(5.0, a[0][0]);
    }
     
    #[test]
    fn index_returns_correct_value() {
        let a = Tensor::of(&[
            &[1.0, 2.0],
            &[3.0, 4.0]
        ]);

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
        let a = Tensor::fill(2, 2, 2.0);
        let b = Tensor::fill(2, 2, 1.0);

        let expected = Tensor::fill(2, 2, 3.0);

        assert_eq!(expected, a + b);
    }

    #[test]
    fn sub_returns_elementwise_diff() {
        let a = Tensor::fill(2, 2, 2.0);
        let b = Tensor::fill(2, 2, 3.0);

        let expected = Tensor::fill(2, 2, -1.0);

        assert_eq!(expected, a - b);
    }

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
    
    #[test]
    fn pow_gives_correct_value() {
        let a = Tensor::of(&[
            &[1.0, 2.0],
            &[3.0, 4.0]
        ]);
        
        let expected = Tensor::of(&[
            &[1.0, 4.0],
            &[9.0, 16.0]
        ]);

        assert_eq!(expected, a.pow(2 as i32));
    }

}
