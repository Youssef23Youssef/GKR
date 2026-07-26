use ark_bn254::Fr;
use ark_ff::{Field, PrimeField, Zero, One, BigInteger};

pub type F = Fr;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_then_subtract_returns_original() {
        // test: (a + b) - b = a 
        let a = F::from(7u64);
        let b = F::from(11u64);

        let result = (a + b) - b;

        assert_eq!(a, result);
        print!("result is {} and a is: {}", result, a);
    }

    #[test]
    fn multiplying_by_one_returns_original() {
        // test: a * 1 = a
        let a = F::from(7u64);
        let result = a * F::one();

        assert_eq!(a, result);
        print!("result is {}", result);
    }

    #[test]
    fn adding_zero_returns_original() {
        // test: a + 0 = a
        let a = F::from(7u64);
        let result = a + F::zero();

        assert_eq!(a, result);
        print!("result is {}", result);
    }

    #[test]
    fn multiplying_by_zero_returns_zero() {
        // test: a * 0 = 0
        let a = F::from(7u64);
        let result = a * F::zero();

        assert_eq!(result, F::zero());
        print!("result is {}", result);
    }

    #[test]
    fn nonzero_element_times_inverse_is_one() {
        // test: a * inverse(a) = 1 when a != 0
        let a = F::from(7u64);
        let a_inv = a.inverse().expect("nonzero elements must have an inverse");

        assert_eq!(a * a_inv, F::one());
    }

    #[test]
    fn inverse_of_zero_is_none() {
        // test: inverse(0) = None
        let zero = F::zero();
        let inverse = zero.inverse();

        assert!(inverse.is_none());
    }

    #[test]
    fn multiplication_distributes_over_addition() {
        // test: a * (b + c) = a*b + a*c
        let a = F::from(7u64);
        let b = F::from(11u64);
        let c = F::from(15u64);

        let lhs = a * (b + c);
        let rhs = a*b + a*c;

        assert_eq!(lhs, rhs);
        print!("lhs is {} and rhs is: {}", lhs, rhs);
    }

    #[test]
    fn large_bytes_reduce_into_field() {
        // test: large byte input is reduced modulo field order
        let mut bytes = F::MODULUS.to_bytes_le();

        // Turn modulus `p` into `p + 5`.
        let mut carry = 5u16;
        for byte in &mut bytes {
            let sum = *byte as u16 + carry;
            *byte = sum as u8;
            carry = sum >> 8;

            if carry == 0 {
                break;
            }
        }

        if carry > 0 {
            bytes.push(carry as u8);
        }

        let reduced = F::from_le_bytes_mod_order(&bytes);

        assert_eq!(reduced, F::from(5u64));
    }
}