use ndarray::ArrayViewMut1;

#[derive(Clone, Debug, Copy)]
pub enum Loss {
    Linear,
    Huber(f64),
}

impl Loss {
    pub fn huber(scaling_factor: f64) -> Self {
        Self::Huber(scaling_factor)
    }

    pub fn apply(&self, res: &mut ArrayViewMut1<f64>) {
        match self {
            Self::Linear => (),
            Self::Huber(s) => {
                let s2 = s * s;
                let s2_inv = 1.0 / s2;
                res.mapv_inplace(|ri| {
                    if ri * ri * s2_inv <= 1.0 {
                        ri
                    } else {
                        2.0 * (ri * ri * s2_inv).sqrt() - 1.0
                    }
                })
            }
        }
    }
}