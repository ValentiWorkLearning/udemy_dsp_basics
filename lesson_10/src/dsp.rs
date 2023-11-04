use crate::impulse_response;
use float_cmp::{*};


pub fn compute_signal_mean(signal_array: &[f64]) -> f64 {
    let mut mean: f64 = Default::default();

    for item in signal_array.iter() {
        mean += item;
    }

    mean / signal_array.len() as f64
}

pub fn compute_signal_variance(signal_array: &[f64]) -> f64 {
    let signal_mean = compute_signal_mean(signal_array);
    let mut signal_variance: f64 = Default::default();

    for sample in signal_array.iter() {
        signal_variance += f64::powi(sample - signal_mean, 2);
    }
    signal_variance / signal_array.len() as f64
}

pub fn compute_signal_devication(variance: f64) -> f64 {
    f64::sqrt(variance)
}

pub fn convolution(signal_array: &[f64], impulse_response:&[f64])->Vec<f64>{
    
    // 1. Generate output signal array according to the convolution equation:
    let mut convolution_result:Vec<f64> = (0..signal_array.len() + impulse_response.len()).map(|_x|{0 as f64}).collect();

    // 2. Step over the source signal
    for i  in 0 .. signal_array.len(){
        let signal_sample = signal_array[i];
        // 3. Convolute with the impulse response
        for j  in 0.. impulse_response.len(){
            let impulse_step = impulse_response[j];
            convolution_result[i+j] = convolution_result[i+j] + signal_sample * impulse_step;
        }
    }
    convolution_result
}

pub fn running_sum(signal_in: &[f64])->Vec<f64>{
    let mut out_signal:Vec<f64> = Vec::new();
    out_signal.push(signal_in[0]);
    for i in 1 .. signal_in.len(){
        out_signal.push(out_signal[i - 1] + signal_in[i]);
    }
    out_signal
}

pub struct PolarSample{
    pub mag:f64,
    pub phase:f64
}

pub struct Polar
{
    pub polar_data:Vec<PolarSample>
}

pub trait DftResult{
    fn as_polar(&self)->Polar;
}

pub trait PolarResult{
    fn as_rectangular(&self)->DftData;
}

impl PolarResult for Polar{
    
    fn as_rectangular(&self)->DftData{
        let mut dft_result = DftData { real_part: Vec::new(), im_part: Vec::new() };
        for sample in &self.polar_data{
            dft_result.real_part.push(sample.mag * sample.phase.cos());
            dft_result.im_part.push(sample.mag*sample.phase.sin());
        }
    
        dft_result
    }
}
pub struct DftData
{
    pub real_part:Vec<f64>,
    pub im_part:Vec<f64>
}
impl DftResult for DftData{
    fn as_polar(&self)->Polar {
        let mut polar_rerp = Polar{
            polar_data:Vec::new()
        };

        for i in 0..self.real_part.len(){
            let mut sample = PolarSample{
                mag: 0.0,
                phase:0.0
            };

            let mut self_real = self.real_part[i];
            let self_im = self.im_part[i];

            sample.mag = (self_real.powf(2.0) + self_im.powf(2.0)).sqrt();

            // Nuisance 2 from http://www.dspguide.com/ch8/9.htm . Correct real part to a small number.
            if float_cmp::approx_eq!(f64,self_real,0.0, ulps=2) {
                self_real = 10.0_f64.powf(-20.0);
            }
            sample.phase = (self_im/self_real).atan();

            // Nuisance 3 correction for phase
            let is_nuisance3_both_negative = (self_real< 0.0) && (self_im < 0.0);
            let is_nuisance3_only_real_negative = self_real < 0.0 &&
                ((self_im > 0.0)|| float_cmp::approx_eq!(f64,self_im,0.0));


            if is_nuisance3_both_negative {
                sample.phase-=std::f64::consts::PI;
            }
            
            if is_nuisance3_only_real_negative{
                sample.phase+=std::f64::consts::PI;
            }
            
            
            polar_rerp.polar_data.push(
                sample
            );
        }

        polar_rerp
    }
}
pub fn dft_transform(signal_array: &[f64])->DftData{
    
    let mut dft_result = DftData
    {
        real_part:(0..signal_array.len() / 2 ).map(|_x|{0 as f64}).collect(),
        im_part: (0..signal_array.len() / 2 ).map(|_x|{0 as f64}).collect()
    };

    // according to http://www.dspguide.com/ch8/6.htm
    let N = signal_array.len();

    for k in 0.. signal_array.len()/2{
        for i in 0.. signal_array.len() - 1{
            let common_part = (2.0* std::f64::consts::PI*(k as f64) *(i as f64) / (N as f64) ) as f64;

            dft_result.real_part[k]=dft_result.real_part[k] + signal_array[i] *  common_part.cos();
            dft_result.im_part[k] = dft_result.im_part[k] - signal_array[i] *  common_part.sin();
        }
    }
    dft_result
}
pub fn compute_signal_magnitude(dft_result:&DftData, signal_length:usize)->Vec<f64>{
    let mut output_mag:Vec<f64> = (0..signal_length/2).map(|_x|{0 as f64}).collect();
    for i in 0..signal_length / 2{
        output_mag[i] = (dft_result.real_part[i].powf(2.0) + dft_result.im_part[i].powf(2.0)).sqrt();
    }
    output_mag
}


pub fn inverse_dft_transform(signal_length:usize, real_part:&[f64], im_part:&[f64])->Vec<f64>{
    let mut output_signal:Vec<f64> = (0..signal_length).map(|_x|{0 as f64}).collect();
    let mut output_rex:Vec<f64> = Vec::new();
    let mut output_imx:Vec<f64> = Vec::new();

    for k in 0..signal_length/2{
        output_rex.push(real_part[k] / signal_length as f64 / 2.0);
        output_imx.push( -im_part[k] / signal_length as f64 / 2.0);
    }

    let N = signal_length;

    for k in 0..real_part.len(){
        for i in 0..signal_length{
            let mut common_multiplier = 2.0* std::f64::consts::PI*(k as f64) *(i as f64) / (N as f64);
            output_signal[i] = output_signal[i] +  real_part[k] * common_multiplier.cos() as f64;
            output_signal[i] = output_signal[i] +  im_part[k] * common_multiplier.sin() as f64;
        }
    }
    output_signal
}