
use crate::dsp;
use crate::dsp::DftResult;

use gnuplot::{*,MultiplotFillOrder::*,MultiplotFillDirection::*};

mod waveforms;
mod impulse_response;

fn gen_signal_vec(arr_to_process:&[f32])->Vec<usize>{
    (0..arr_to_process.len()).map(|x| x).collect()
}


pub fn draw_convolution_sample(){

    let mut fg = Figure::new();
	fg.set_multiplot_layout(2, 2)
		.set_title("Signal and Inpulse response")
		.set_scale(0.8, 0.8)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);

    let signal_vec = gen_signal_vec(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ);
    let impuse_vec = gen_signal_vec(&impulse_response::IMPULSE_RESPONSE);

	fg.axes2d().lines(
		&signal_vec,
		&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ,
		&[Caption("Signal"),Color("black")],
	);
    
    fg.axes2d().lines(
		&impuse_vec,
		&impulse_response::IMPULSE_RESPONSE,
		&[Caption("Impulse response"),Color("red")],
	);
    


    let convolution_out_side = dsp::convolution_output_side(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ, &impulse_response::IMPULSE_RESPONSE);
    let convolution_out_side_vec = gen_signal_vec(&convolution_out_side);
    fg.axes2d().lines(
		&convolution_out_side_vec,
		&convolution_out_side,
		&[Caption("Convolution Output side"),Color("black")],
	);



    let convolution_result = dsp::convolution(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ, &impulse_response::IMPULSE_RESPONSE);
    let convolution_vec = gen_signal_vec(&convolution_result);
    fg.axes2d().lines(
		&convolution_vec,
		&convolution_result,
		&[Caption("Convolution"),Color("blue")],
	);


    // let running_sum_result = dsp::running_sum(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ);
    // let running_sum_vec = gen_signal_vec(&running_sum_result);

    // fg.axes2d().lines(
	// 	&running_sum_vec,
	// 	&running_sum_result,
	// 	&[Caption("Running sum"),Color("blue")],
	// );

	fg.show().unwrap();
}

pub fn draw_dft_sample(){
    let mut fg = Figure::new();
	fg.set_multiplot_layout(2, 2)
		.set_title("Signal and DFT representation")
		.set_scale(0.8, 0.8)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);

    let signal_vec = gen_signal_vec(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ);

	fg.axes2d().lines(
		&signal_vec,
		&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ,
		&[Caption("Signal"),Color("black")],
	);
    

    let dft_result = dsp::dft_transform(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ);
    let real_vec = gen_signal_vec(&dft_result.real_part);
    let im_vec = gen_signal_vec(&dft_result.im_part);

    fg.axes2d().lines(
		&real_vec,
		&dft_result.real_part,
		&[Caption("Real Part"),Color("blue")],
	);

    fg.axes2d().lines(
		&im_vec,
		&dft_result.im_part,
		&[Caption("Imaginary part"),Color("blue")],
	);


    let idft_result = dsp::inverse_dft_transform(
        waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ.len(),
        &dft_result.real_part,
        &dft_result.im_part
    );
    let idft_vec = gen_signal_vec(&idft_result);

    fg.axes2d().lines(
		&idft_vec,
		&idft_result,
		&[Caption("Synthesis from IDFT"),Color("red")],
	);


	fg.show().unwrap();
}

pub fn draw_fft_over_ecg()
{
    let mut fg = Figure::new();
	fg.set_multiplot_layout(2, 2)
		.set_title("ECG Signal and DFT representation")
		.set_scale(0.8, 0.8)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);

    let signal_vec = gen_signal_vec(&waveforms::ECG_SIGNAL);


    let dft_result = dsp::dft_transform(&waveforms::ECG_SIGNAL);

    fg.axes2d().lines(
		&signal_vec,
		&waveforms::ECG_SIGNAL,
		&[Caption("ECG Signal"),Color("red")],
	);
    
    let signal_mag = dsp::compute_signal_magnitude(&dft_result,waveforms::ECG_SIGNAL.len());
    let mag_vec = gen_signal_vec(&signal_mag);

    fg.axes2d().lines(
		&mag_vec,
		&signal_mag,
		&[Caption("Signal magnitude"),Color("black")],
	);

    let real_vec = gen_signal_vec(&dft_result.real_part);
    let im_vec = gen_signal_vec(&dft_result.im_part);


    fg.axes2d().lines(
		&real_vec,
		&dft_result.real_part,
		&[Caption("Real Part"),Color("blue")],
	);

    fg.axes2d().lines(
		&im_vec,
		&dft_result.im_part,
		&[Caption("Imaginary part"),Color("blue")],
	);

    fg.show().unwrap();
}

pub fn draw_rectangular_to_polar_sample(){

    let mut fg = Figure::new();
	fg.set_multiplot_layout(2, 2)
		.set_title("DFT representation with polar notation")
		.set_scale(1.2, 1.2)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);

    let dft_result = dsp::dft_transform(&waveforms::ECG_SIGNAL);
    let real_vec = gen_signal_vec(&dft_result.real_part);
    let im_vec = gen_signal_vec(&dft_result.im_part);


    fg.axes2d().lines(
		&real_vec,
		&dft_result.real_part,
		&[Caption("Real Part"),Color("blue")],
	);

    fg.axes2d().lines(
		&im_vec,
		&dft_result.im_part,
		&[Caption("Imaginary part"),Color("red")],
	);


    let polar_notation =  dft_result.as_polar();
    let extracted_magnitude : Vec<f32> = polar_notation.polar_data.iter().map(|sample| sample.mag ).collect();
    let magnitude_x = gen_signal_vec(&extracted_magnitude);


    let extracted_phase : Vec<f32> = polar_notation.polar_data.iter().map(|sample| sample.phase ).collect();
    let phase_x = gen_signal_vec(&extracted_phase);

    fg.axes2d().lines(
		&magnitude_x,
		&extracted_magnitude,
		&[Caption("Magnitude"),Color("green")],
	);

    fg.axes2d().lines(
		&phase_x,
		&extracted_phase,
		&[Caption("Phase"),Color("black")],
	);

    fg.show().unwrap();
}

pub fn draw_20khz_rex_imx_sample_with_complex_dft(){
    let mut fg = Figure::new();
	fg.set_multiplot_layout(2, 2)
		.set_title("Signal real and imaginary parts")
		.set_scale(0.8, 0.8)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);

    let signal_vec_rex_imx = gen_signal_vec(&waveforms::SIG_20_HZ_REX);

	fg.axes2d().lines(
		&signal_vec_rex_imx,
		&waveforms::SIG_20_HZ_REX,
		&[Caption("Real part"),Color("black")],
	);
    
    fg.axes2d().lines(
		&signal_vec_rex_imx,
		&waveforms::SIH_20_HZ_IMX,
		&[Caption("Imaginary part"),Color("black")],
	);

    let signal_complex_dft = dsp::complex_dft_transform(&waveforms::SIG_20_HZ_REX,&waveforms::SIH_20_HZ_IMX);
    
    
    let extracted_real : Vec<f32> = signal_complex_dft.iter().map(|sample| sample.re ).collect();
    let real_x = gen_signal_vec(&extracted_real);

    fg.axes2d().lines(
		&real_x,
		&extracted_real,
		&[Caption("DFT Real"),Color("red")],
	);


    let extracted_imagine : Vec<f32> = signal_complex_dft.iter().map(|sample| sample.im ).collect();
    let im_x = gen_signal_vec(&extracted_imagine);

    fg.axes2d().lines(
		&im_x,
		&extracted_imagine,
		&[Caption("DFT Imagine"),Color("blue")],
	);


    fg.show().unwrap();
}

pub fn draw_fft_vs_dft()
{
    let mut fg = Figure::new();
	fg.set_multiplot_layout(2, 2)
		.set_title("DFT vs FFT")
		.set_scale(0.8, 0.8)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);


    let test_sequence_real:[f32; 8] = [0.46,0.72,-0.3,-0.09,-0.16,-0.2,0.0, -0.43 ];
    let test_sequence_imagine:[f32; 8] = [ 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0];
    let fft_result = dsp::fft_transform(&test_sequence_real);
    let dft_result = dsp::complex_dft_transform(&test_sequence_real,&test_sequence_imagine);

    {
    
        let extracted_real : Vec<f32> = dft_result.iter().map(|sample| sample.re ).collect();
        let real_x = gen_signal_vec(&extracted_real);
    
        fg.axes2d().lines(
            &real_x,
            &extracted_real,
            &[Caption("DFT Real"),Color("red")],
        );
    
    
        let extracted_imagine : Vec<f32> = dft_result.iter().map(|sample| sample.im ).collect();
        let im_x = gen_signal_vec(&extracted_imagine);
    
        fg.axes2d().lines(
            &im_x,
            &extracted_imagine,
            &[Caption("DFT Imagine"),Color("red")],
        );
    }

    {
        let extracted_real_fft : Vec<f32> = fft_result.iter().map(|sample| sample.re ).collect();
        let real_x = gen_signal_vec(&extracted_real_fft);
    
        fg.axes2d().lines(
            &real_x,
            &extracted_real_fft,
            &[Caption("FFT Real"),Color("blue")],
        );
    
    
        let extracted_imagine_fft : Vec<f32> = fft_result.iter().map(|sample| sample.im ).collect();
        let im_x = gen_signal_vec(&extracted_imagine_fft);
    
        fg.axes2d().lines(
            &im_x,
            &extracted_imagine_fft,
            &[Caption("FFT Imagine"),Color("blue")],
        );
    }

    fg.show().unwrap();
}


pub fn draw_hamming_blackman_windows(){
    let mut fg = Figure::new();
	fg.set_multiplot_layout(2, 2)
		.set_title("Hamming and Blackman windows")
		.set_scale(0.8, 0.8)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);

    let hamming_window : Vec<f32> = dsp::compute_hamming_window(128);
    let hamming_window_x = gen_signal_vec(&hamming_window);

    fg.axes2d().lines(
        &hamming_window_x,
        &hamming_window,
        &[Caption("Hamming window"),Color("red")],
    );

    let blackman_window : Vec<f32> = dsp::compute_blackman_window(128);
    let blackman_window_x = gen_signal_vec(&blackman_window);

    fg.axes2d().lines(
        &blackman_window_x,
        &blackman_window,
        &[Caption("Blackman window"),Color("red")],
    );
    
    fg.show().unwrap();
}

pub fn draw_designed_filter_sample(){
    let mut fg = Figure::new();
	fg.set_multiplot_layout(2, 2)
		.set_title("Designed low-pass filter demo")
		.set_scale(0.8, 0.8)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);

    let filter_kernel_lpf = dsp::design_windowed_sinc_filter(10000.0, 48000.0, 59);
    let filtred_result_lpf = dsp::convolution(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ, &filter_kernel_lpf);

    let input_signal_vec = gen_signal_vec(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ);
    let filtred_signal_vec = gen_signal_vec(&filtred_result_lpf);

	fg.axes2d().lines(
		&input_signal_vec,
		&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ,
		&[Caption("Signal"),Color("black")],
	);
    
    fg.axes2d().lines(
		&filtred_signal_vec,
		&filtred_result_lpf,
		&[Caption("Filtred signal LPF"),Color("red")],
	);

    let filter_kernel_hpf = dsp::design_windowed_sinc_filter_hpf(1000.0, 48000.0,28);
    
    let filtred_result_hpf = dsp::convolution(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ, &filter_kernel_hpf);
    fg.axes2d().lines(
		&filtred_signal_vec,
		&filtred_result_hpf,
		&[Caption("Filtred signal HPF"),Color("red")],
	);

    let hpf_kernel_vec = gen_signal_vec(&filter_kernel_hpf);
    fg.axes2d().lines(
		&hpf_kernel_vec,
		&filter_kernel_hpf,
		&[Caption("Impulse response HPF"),Color("red")],
	);

    fg.show().unwrap();
}


pub fn draw_hpf_hpf_impulse_step_response(){
	let mut fg = Figure::new();
	fg.set_multiplot_layout(2, 2)
		.set_title("Designed low-pass filter demo")
		.set_scale(0.8, 0.8)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);

    let filter_kernel_lpf = dsp::design_windowed_sinc_filter_hpf(10000.0, 48000.0, 59);
   
   	let mut filter_kernel_padded = filter_kernel_lpf.clone();
	while filter_kernel_padded.len() != 64 {
		filter_kernel_padded.push(0.0_f32);
	}

    let frequency_response = dsp::fft_transform(&filter_kernel_padded);
	let fft_magnitude:Vec<f32>= frequency_response
		.iter()
		.enumerate()
		.filter(|&(i,_)|
			{ return i<= frequency_response.len() / 2;
			})
		.map(|(_,complex_result)|
		{
			return (complex_result.re.powf(2.0_f32) + complex_result.im.powf(2.0_f32)).sqrt() as f32;
		}).collect();

	let filter_kernel_vec = gen_signal_vec(&filter_kernel_lpf);
	let fft_result = gen_signal_vec(&fft_magnitude);


	fg.axes2d().lines(
		&filter_kernel_vec,
		&filter_kernel_lpf,
		&[Caption("LPF window"),Color("black")],
	);
    
    fg.axes2d().lines(
		&fft_result,
		&fft_magnitude,
		&[Caption("FFT over impulse response"),Color("blue")],
	);

	fg.show().unwrap();

}

pub fn draw_designed_bandpass_filter(){
    let mut fg = Figure::new();
	fg.set_multiplot_layout(4, 2)
		.set_title("Designed bandpass filter demo")
		.set_scale(1.0, 1.0)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);

    let filter_kernel_highpass = dsp::design_bandbass_filter(12000.0,16000.0, 48000.0, 50);
    let filter_kernel_lowpass = dsp::design_bandbass_filter(200.0,1500.0, 48000.0, 50);
    let filtred_result_hpf = dsp::convolution(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ, &filter_kernel_highpass);
    let filtred_result_lpf = dsp::convolution(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ, &filter_kernel_lowpass);

    let input_signal_vec = gen_signal_vec(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ);
    let filtred_signal_vec = gen_signal_vec(&filtred_result_hpf);

    let filter_kernel_vec = gen_signal_vec(&filter_kernel_highpass);

    fg.axes2d().lines(
		&filter_kernel_vec,
		&filter_kernel_highpass,
		&[Caption("Filter kernel"),Color("black")],
	);
    
    fg.axes2d().lines(
		&filtred_signal_vec,
		&filtred_result_hpf,
		&[Caption("Filtred HPF"),Color("red")],
	);
    fg.axes2d().lines(
		&filtred_signal_vec,
		&filtred_result_lpf,
		&[Caption("Filtred LPF"),Color("red")],
	);

	fg.axes2d().lines(
		&input_signal_vec,
		&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ,
		&[Caption("Signal"),Color("black")],
	);


    let dft_result_hpf = dsp::dft_transform(&filtred_result_hpf);
    
    let signal_mag = dsp::compute_signal_magnitude(&dft_result_hpf,filtred_result_hpf.len());
    let mag_vec = gen_signal_vec(&signal_mag);

    fg.axes2d().lines(
		&mag_vec,
		&signal_mag,
		&[Caption("HPF magnitude"),Color("black")],
	);


    let dft_result_lpf = dsp::dft_transform(&filtred_result_lpf);
    
    let signal_mag_lpf = dsp::compute_signal_magnitude(&dft_result_lpf,filtred_result_lpf.len());
    let mag_vec_lpf = gen_signal_vec(&signal_mag_lpf);

    fg.axes2d().lines(
		&mag_vec_lpf,
		&signal_mag_lpf,
		&[Caption("LPF magnitude"),Color("black")],
	);

    fg.show().unwrap();
}



pub fn draw_amplitute_modulation_sample(){
    let mut fg = Figure::new();
	fg.set_multiplot_layout(4, 2)
		.set_title("Amplitude modulation demo")
		.set_scale(1.0, 1.0)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);


    const NUM_SAMPLES_PER_WAVE:u32 = 1024;
    const CARRIER_FREQUENCY:u32 = 20;
    const VOICE_TONE:u32 = 2;

    let mut carrier_signal_array:Vec<f32> = Vec::new();
    let mut voice_signal:Vec<f32> = Vec::new();

    // Chapter 10 - Fourier Transform Properties 205, Amplitude modulation

    for i in 0..NUM_SAMPLES_PER_WAVE{
        let carrier_sample = (2.0_f32 * std::f32::consts::PI * CARRIER_FREQUENCY as f32 * i as f32 / NUM_SAMPLES_PER_WAVE as f32).sin();
        let voice_sample = (2.0_f32 * std::f32::consts::PI * VOICE_TONE as f32 * i as f32 / NUM_SAMPLES_PER_WAVE as f32).sin();
        carrier_signal_array.push(carrier_sample);
        voice_signal.push(voice_sample);
    }

    let carrier_x = gen_signal_vec(&carrier_signal_array);

    fg.axes2d().lines(
		&carrier_x,
		&carrier_signal_array,
		&[Caption("Carrier signal"),Color("black")],
	);
    fg.axes2d().lines(
		&carrier_x,
		&voice_signal,
		&[Caption("Voice simulated signal"),Color("black")],
	);

    carrier_signal_array.iter_mut().enumerate().for_each(move |(sample_index,carrier_sample)| { *carrier_sample*=voice_signal[sample_index] });


    fg.axes2d().lines(
		&carrier_x,
		&carrier_signal_array,
		&[Caption("AM Modulated signal"),Color("blue")],
	);

    let fft = dsp::fft_transform(&carrier_signal_array);

    let fft_magnitude : Vec<f32> = fft.iter().map(|sample| (sample.re.powf(2.0) + sample.im.powf(2.0)).sqrt() ).collect();
    let fft_magnitute_x = gen_signal_vec(&fft_magnitude);

    fg.axes2d().lines(
            &fft_magnitute_x,
            &fft_magnitude,
            &[Caption("FFT magnitude"),Color("red")],
        );
    fg.show().unwrap();
}