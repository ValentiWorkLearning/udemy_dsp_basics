use float_cmp::*;
use num::{complex::Complex, traits::Pow};

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


pub fn convolution_output_side(signal_array: &[f64], impulse_response:&[f64])->Vec<f64>{
        // 1. Generate output signal array according to the convolution equation:
    let mut convolution_result:Vec<f64> = (0..signal_array.len() + impulse_response.len()).map(|_x|{0 as f64}).collect();
    for i  in 0..convolution_result.len(){
        for j in 0.. impulse_response.len(){
            if (i as i128 - j as i128 ) < 0{
                continue;
            }
            if i+j >= signal_array.len() {
                continue;
            }
            //println!("I value: {}, J value:{}", i,j);
            convolution_result[i] = convolution_result[i] + impulse_response[j] * signal_array[i-j];
            
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
    let n = signal_array.len();

    for k in 0.. signal_array.len()/2{
        for i in 0.. signal_array.len() - 1{
            let common_part = (2.0* std::f64::consts::PI*(k as f64) *(i as f64) / (n as f64) ) as f64;

            dft_result.real_part[k] = dft_result.real_part[k] + signal_array[i] *  common_part.cos();
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

    let n = signal_length;

    for k in 0..real_part.len(){
        for i in 0..signal_length{
            let common_multiplier = 2.0* std::f64::consts::PI*(k as f64) *(i as f64) / (n as f64);
            output_signal[i] = output_signal[i] +  real_part[k] * common_multiplier.cos() as f64;
            output_signal[i] = output_signal[i] +  im_part[k] * common_multiplier.sin() as f64;
        }
    }
    output_signal
}

pub fn complex_dft_transform(signal_array_real: &[f64],signal_array_im: &[f64])->Vec<num::complex::Complex64>{
    let mut output_complex:Vec<num::complex::Complex64> = (0..signal_array_real.len()).map(|_x|{Complex::new(0.0,0.0)}).collect();

    // Frequency domain
    for k in 0..output_complex.len(){
        // Time domain
        for i in 0..signal_array_real.len(){
            // From http://www.dspguide.com/ch12/3.htm
            let sr = (2.0*std::f64::consts::PI*k as f64 *i as f64 / signal_array_real.len() as f64).cos();
            let si = -(2.0*std::f64::consts::PI*k as f64 *i as f64 / signal_array_real.len() as f64 ).sin();
            // From Euler's formuale:http://embeddedsystemengineering.blogspot.com/2016/06/complex-dft-and-fft-algorithm.html

            let real_part = signal_array_real[i] * sr - signal_array_im[i]* si;
            let im_part = signal_array_real[i] * si + signal_array_im[i] * sr;
            //https://www.udemy.com/course/digital-signal-processing-dsp-from-ground-uptm-in-c/learn/lecture/11368702#overview 
            // The author didn't get properly the formulae from DSPguide
            //let im_part = signal_array_im[i] * si - signal_array_im[i] * sr;

            output_complex[k] += Complex::new( real_part,im_part);
        }
    }
    output_complex
}

fn compute_bit_reverse_index(index:usize, order:u8)->usize{
    let mut result:usize  = 0;
    let mut order_count:u8 = order;
    let mut index_local = index;

    while order_count!=0 {
        result <<= 1;
        result |= index_local & 1;
        index_local = index_local >> 1;
        order_count-=1;
    }
    result
}


fn compute_fft_stages_order(signal_array:&[f64])->u8{
    let mut pow_order:u8 = 0;
    let mut buf_size = signal_array.len();
    while buf_size % 2 == 0 {
        buf_size /= 2;
        pow_order+=1;
    }
    pow_order
}


fn bit_reversal_in_place(signal_array:&[f64])->Vec<num::complex::Complex64>{
    let mut bit_reversed:Vec<num::complex::Complex64> = signal_array.iter().map(|real_sample|{ num::complex::Complex64::new(*real_sample, 0.0_f64) }).collect();

    let order = compute_fft_stages_order(signal_array);
    let num_iterations = signal_array.len() / 2;
    for i  in 0 .. num_iterations {
        let current_index = i;
        let reversed_index = compute_bit_reverse_index(current_index, order);
        bit_reversed.swap(current_index, reversed_index);
    }

    bit_reversed
}

trait TwindleFactorOrdered {
    fn compute_for_order(&self)->num::complex::Complex64;
}
struct TwindleFactor{
    index:u8,
    order:u8
}

impl TwindleFactorOrdered for TwindleFactor{
    fn compute_for_order(&self)->num::complex::Complex64 {
        let real:f64 = (2.0_f64 * std::f64::consts::PI * self.index as f64 / self.order as f64).cos();
        let imag:f64 = -(2.0_f64 * std::f64::consts::PI * self.index as f64 / self.order as f64).sin();

        num::complex::Complex64::new(real,imag)
    }
}

pub fn fft_transform(signal_array:&[f64])->Vec<num::complex::Complex64>{
    // Implemented according to https://www.linkedin.com/pulse/how-fft-algorithm-works-part-1-repeating-mark-newman/
    // and
    // https://www.dspguide.com/ch12.htm
    let mut output_vector = bit_reversal_in_place(&signal_array);

    let num_stages = compute_fft_stages_order(&signal_array);

    for i in 1 ..num_stages+1{
        let step = 2_usize.pow(i as u32);
        let internal_it = step / 2;

        for j in (0..output_vector.len()).step_by(step){
            for k in 0 .. internal_it{
                let even_sample = output_vector[j + k];
                let odd_sample = output_vector[j + k + internal_it] * TwindleFactor{ index: k as u8, order: step as u8 }.compute_for_order();
                let even_result = even_sample + odd_sample;
                let odd_result = even_sample - odd_sample;
                output_vector[j + k] = even_result;
                output_vector[j + k + internal_it] = odd_result;
            }
        }
    }
    output_vector
}



pub fn compute_hamming_window(filter_kernel_length:usize)->Vec<f64>{
    let mut filter_kernel:Vec<f64> = (0..filter_kernel_length).map(|_x|{0 as f64}).collect();
    for i in 0.. filter_kernel_length{
            let computed_sample = 2.0_f64*std::f64::consts::PI * i as f64 / filter_kernel_length as f64;
            filter_kernel[i] = 0.54_f64 -0.46_f64*computed_sample.cos();
    }
    filter_kernel
}

pub fn compute_blackman_window(filter_kernel_length:usize)->Vec<f64>{
    let mut filter_kernel:Vec<f64> = (0..filter_kernel_length).map(|_x|{0 as f64}).collect();
    for i in 0.. filter_kernel_length{
        let computed_sample = 2.0_f64*std::f64::consts::PI * i as f64 / filter_kernel_length as f64;
        let second_sample = 4.0_f64*std::f64::consts::PI * i as f64 / filter_kernel_length as f64;
        filter_kernel[i] = 0.42_f64 -0.5_f64*computed_sample.cos() + 0.08_f64*second_sample.cos();
    }
    filter_kernel
}


fn normalize_frequency(cutoff_frequency:f64,sampling_frequency:f64)->f64{
    let nyquist_frequency = sampling_frequency / 2.0_f64;
    // Propotion example
    // nyqust_frequency: 0.5 = 24'000
    // Cutoff frequency: x = 10'000
    // x = 10'000/24'000 * 0.5
    // normalized_cutoff_frequency = cutoff_frequency/nyquist_frequency * 0.5

    let normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency * 0.5;
    normalized_cutoff_frequency
}

pub fn design_windowed_sinc_filter(cutoff_frequency:f64,sampling_frequency:f64, filter_kernel_length:usize)->Vec<f64>{
    let mut filter_kernel:Vec<f64> = (0..filter_kernel_length).map(|_x|{0 as f64}).collect();

    let k: f64 = 1.0_f64;
    let normalized_cutoff_frequency = normalize_frequency(cutoff_frequency,sampling_frequency);
    let hamming_window = compute_blackman_window(filter_kernel_length);

    // http://www.dspguide.com/ch16/2.htm
    for i in 0..filter_kernel_length{
        if i  == (filter_kernel_length / 2){
            filter_kernel[i] = 2.0_f64*std::f64::consts::PI*normalized_cutoff_frequency*k;
        }
        else{
            let sinc_base = 2.0_f64*std::f64::consts::PI*normalized_cutoff_frequency*(i as f64 - filter_kernel_length as f64/2.0_f64) / ( i  as f64 - filter_kernel_length as f64/2.0_f64);
            filter_kernel[i] = k * sinc_base;
            filter_kernel[i]*=hamming_window[i];
        }
    }
    let normalization_factor_sum:f64 = filter_kernel.iter().sum();
    filter_kernel.iter_mut().for_each(|koef| *koef/= normalization_factor_sum);

    filter_kernel
}


pub fn perform_spectral_reversal(filter_kernel:&mut Vec<f64>){
    for i in 0..filter_kernel.len(){
        filter_kernel[i] = (-1.0_f64).powf(i as f64) * filter_kernel[i];
    }
}

pub fn perform_spectral_inversion(filter_kernel:&mut Vec<f64>){
    for i in 0..filter_kernel.len(){
        filter_kernel[i] = -filter_kernel[i];
    }
    let kernel_length = filter_kernel.len() / 2;
    filter_kernel[kernel_length] +=1.0;
}
pub fn design_windowed_sinc_filter_hpf(cutoff_frequency:f64,sampling_frequency:f64, filter_kernel_length:usize)->Vec<f64>{
    let mut kernel = design_windowed_sinc_filter(cutoff_frequency,sampling_frequency,filter_kernel_length);
    perform_spectral_inversion(&mut kernel);
    kernel
}
pub fn design_bandbass_filter(low_bandpass_freq:f64,high_bandpass_freq:f64,sampling_frequency:f64, filter_kernel_length:usize)->Vec<f64>{
    let mut low_passband_filter_kernel:Vec<f64> = design_windowed_sinc_filter(low_bandpass_freq,sampling_frequency,filter_kernel_length);
    let mut high_passband_filter_kernel:Vec<f64> = design_windowed_sinc_filter(high_bandpass_freq, sampling_frequency, filter_kernel_length);

    let mut filter_kernel:Vec<f64> = (0..filter_kernel_length).map(|_x|{0 as f64}).collect();

    // Apply spectrum inversion to the high pass filter
    perform_spectral_inversion(&mut high_passband_filter_kernel);

    // Combine lowpass and highpass filter
    for i in 0..filter_kernel_length{
        filter_kernel[i] = low_passband_filter_kernel[i] + high_passband_filter_kernel[i];
    }
    
    // Apply spectrum inversion to the band-reject filter to obtain the bandpass filter
    perform_spectral_inversion(&mut filter_kernel);
    filter_kernel
}