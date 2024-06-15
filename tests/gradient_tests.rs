#[cfg(test)]
mod gradient_tests {
    use llm_rs::tensor::{Backward, Tensor};

    #[test]
    fn small_computation_graph() {
        // y = f * ((a * b) + c)
        //   = f * (e + c)
        //   = f * d
        let a = Tensor::singleton(1.0).named("a".to_string());
        let b = Tensor::singleton(2.0).named("b".to_string());
        let e = &a * &b;
        let c = Tensor::singleton(10.0).named("c".to_string());
        let d = &e + &c;
        let f = Tensor::singleton(-2.0).named("f".to_string());

        let y = &f * &d;

        // Assert correct value
        assert_eq!(2.0, e.item());
        assert_eq!(12.0, d.item());
        assert_eq!(-24.0, y.item());

        // Propogate gradient
        y.set_grad(1.0);
        y.backward();

        // Assert correct gradient
        // f = -2.0
        // d = e + c
        // y = f * d
        assert_eq!(1.0, y.grad());
        // dL/df = (dL/dy)(dy/df) = y.grad * d.last = 1 * 12 = 12
        assert_eq!(f.item(), d.grad());
        // dL/dd = (dL/dy)(dy/dd) = y.grad * f.last = 1 * -2 = -2
        assert_eq!(d.item(), f.grad());

        // c = 10.0
        // e = a * b
        // d = e + c
        // dL/de = (dL/dd)(dd/de) = dL/dd * 1 = d.grad
        assert_eq!(d.grad(), e.grad());
        // dL/dc = (dL/dy)(dy/dd) = dL/dE * 1 = d.grad
        assert_eq!(d.grad(), c.grad());

        // a = 1.0
        // b = 2.0
        // e = a * b
        // dL/da = (dL/de)(de/da) = e.grad * b.last
        assert_eq!(e.grad() * b.item(), a.grad());
        // dL/db = (dL/de)(de/db) = e.grad * a.last
        assert_eq!(e.grad() * a.item(), b.grad());
    }
}
