#[cfg(test)]
mod gradient_tests {
    use llm_rs::{operations::Differentiable, tensor::Tensor};

    #[test]
    fn small_computation_graph() {
        let a_val = 1.0;
        let b_val = 2.0;
        let c_val = 10.0;
        let f_val = -2.0;
        // y = f * ((a * b) + c)
        //   = f * (e + c)
        //   = f * d
        let a = Tensor::singleton(a_val).named("a".to_string()).with_grad();
        let b = Tensor::singleton(b_val).named("b".to_string()).with_grad();
        let e = &a * &b;
        let c = Tensor::singleton(c_val).named("c".to_string()).with_grad();
        let d = &e + &c;
        let f = Tensor::singleton(f_val).named("f".to_string()).with_grad();

        let y = &f * &d;

        // Assert correct value
        let y_val = y.item();
        assert_eq!(-24.0, y_val);
        let d_val = d.item();
        assert_eq!(12.0, d_val);
        let e_val = e.item();
        assert_eq!(2.0, e_val);

        // Propogate gradient
        y.set_grad(Tensor::singleton(1.0));
        Differentiable::backward(&y);
        // f = -2.0
        // d = e + c
        // ---------
        // y = f * d
        let y_grad = y.grad();
        assert_eq!(1.0, y_grad.item());
        // d.grad = dL/dd = (dL/dy)(dy/dd) = y.grad * f.last = 1 * -2 = -2
        let d_grad = d.grad();
        assert_eq!(d_grad, f.clone() * y.clone().grad());
        assert_eq!(d_grad.item(), -2.0);
        // f.grad = dL/df = (dL/dy)(dy/df) = y.grad * d.last = 1 * 12 = 12
        let f_grad =  f.grad();
        assert_eq!(f_grad, d.clone() * y.clone().grad());
        assert_eq!(f_grad.item(), 12.0);

        // Assert correct gradient

        // c = 10.0
        // e = a * b
        // ---------
        // d = e + c
        // e.grad = dL/de = (dL/dd)(dd/de) = dL/dd * 1 = d.grad = -2
        assert!(e.grad() == d.grad() && e.grad().item() == -2.0);
        // c.grad = dL/dc = (dL/dy)(dy/dd) = dL/dE * 1 = d.grad = -2
        assert!(c.grad() == d.grad() && c.grad().item() == -2.0);

        // a = 1.0
        // b = 2.0
        // ---------
        // e = a * b
        // a.grad = dL/da = (dL/de)(de/da) = e.grad * b.last = -2 * 2 = -4
        assert!(a.grad() == e.grad() * b.clone() && a.grad().item() == -4.0);
        // b.grad = dL/db = (dL/de)(de/db) = e.grad * a.last = -2 * 1 = -2
        assert!(b.grad() == e.grad() * a.clone() && b.grad().item() == -2.0);
        //
    }
}
