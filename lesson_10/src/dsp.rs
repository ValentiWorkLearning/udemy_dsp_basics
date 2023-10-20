use crate::impulse_response;



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
pub struct DftExecutionResult
{
    pub real_part:Vec<f64>,
    pub im_part:Vec<f64>
}

pub fn dft_transform(signal_array: &[f64])->DftExecutionResult{
    
    let mut dft_result = DftExecutionResult
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