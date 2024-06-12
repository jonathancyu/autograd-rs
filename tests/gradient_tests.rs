#[cfg(test)]
mod gradient_tests {

    use llm_rs::tensor::{Backward, Tensor};

    #[test]
    fn small_computation_graph() {
        // y = f * ((a * b) + c)
        //   = f * (e + c)
        //   = f * d
        let a = Tensor::singleton(1.0);
        let b = Tensor::singleton(2.0);
        let e = &a * &b;
        let c = Tensor::singleton(10.0);
        let d = &e + &c;
        let f = Tensor::singleton(-2.0);

        let y = &f * &d;

        assert_eq!(2.0, e.item());
        assert_eq!(12.0, d.item());
        assert_eq!(-24.0, y.item());
        y.backward(1.0);


        // *e.grad.borrow_mut() = 1.0;
        // e.backward(1.0);
        // assert_eq!(0.0, *a.grad.borrow_mut());
        // assert_eq!(0.0, *a.grad.borrow_mut());
    }
}
