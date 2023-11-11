pub mod waveforms;
pub mod dsp;
pub mod impulse_response;
use dsp::DftResult;
use gnuplot::{*,MultiplotFillOrder::*,MultiplotFillDirection::*};


fn gen_signal_vec(arr_to_process:&[f64])->Vec<usize>{
    (0..arr_to_process.len()).map(|x| x).collect()
}


fn draw_convolution_sample(){

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

fn draw_dft_sample(){
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

fn draw_fft_over_ecg()
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

fn draw_rectangular_to_polar_sample(){

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
    let extracted_magnitude : Vec<f64> = polar_notation.polar_data.iter().map(|sample| sample.mag ).collect();
    let magnitude_x = gen_signal_vec(&extracted_magnitude);


    let extracted_phase : Vec<f64> = polar_notation.polar_data.iter().map(|sample| sample.phase ).collect();
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

fn draw_20khz_rex_imx_sample_with_complex_dft(){
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
    
    
    let extracted_real : Vec<f64> = signal_complex_dft.iter().map(|sample| sample.re ).collect();
    let real_x = gen_signal_vec(&extracted_real);

    fg.axes2d().lines(
		&real_x,
		&extracted_real,
		&[Caption("DFT Real"),Color("red")],
	);


    let extracted_imagine : Vec<f64> = signal_complex_dft.iter().map(|sample| sample.im ).collect();
    let im_x = gen_signal_vec(&extracted_imagine);

    fg.axes2d().lines(
		&im_x,
		&extracted_imagine,
		&[Caption("DFT Imagine"),Color("blue")],
	);


    fg.show().unwrap();
}

fn draw_fft_vs_dft()
{
    let mut fg = Figure::new();
	fg.set_multiplot_layout(2, 2)
		.set_title("DFT vs FFT")
		.set_scale(0.8, 0.8)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);


    let test_sequence_real:[f64; 8] = [0.46,0.72,-0.3,-0.09,-0.16,-0.2,0.0, -0.43 ];
    let test_sequence_imagine:[f64; 8] = [ 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0];
    let fft_result = dsp::fft_transform(&test_sequence_real);
    let dft_result = dsp::complex_dft_transform(&test_sequence_real,&test_sequence_imagine);

    {
    
        let extracted_real : Vec<f64> = dft_result.iter().map(|sample| sample.re ).collect();
        let real_x = gen_signal_vec(&extracted_real);
    
        fg.axes2d().lines(
            &real_x,
            &extracted_real,
            &[Caption("DFT Real"),Color("red")],
        );
    
    
        let extracted_imagine : Vec<f64> = dft_result.iter().map(|sample| sample.im ).collect();
        let im_x = gen_signal_vec(&extracted_imagine);
    
        fg.axes2d().lines(
            &im_x,
            &extracted_imagine,
            &[Caption("DFT Imagine"),Color("red")],
        );
    }

    {
        let extracted_real_fft : Vec<f64> = fft_result.iter().map(|sample| sample.re ).collect();
        let real_x = gen_signal_vec(&extracted_real_fft);
    
        fg.axes2d().lines(
            &real_x,
            &extracted_real_fft,
            &[Caption("FFT Real"),Color("blue")],
        );
    
    
        let extracted_imagine_fft : Vec<f64> = fft_result.iter().map(|sample| sample.im ).collect();
        let im_x = gen_signal_vec(&extracted_imagine_fft);
    
        fg.axes2d().lines(
            &im_x,
            &extracted_imagine_fft,
            &[Caption("FFT Imagine"),Color("blue")],
        );
    }

    fg.show().unwrap();
}


fn draw_hamming_blackman_windows(){
    let mut fg = Figure::new();
	fg.set_multiplot_layout(2, 2)
		.set_title("Hamming and Blackman windows")
		.set_scale(0.8, 0.8)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);

    let hamming_window : Vec<f64> = dsp::compute_hamming_window(128);
    let hamming_window_x = gen_signal_vec(&hamming_window);

    fg.axes2d().lines(
        &hamming_window_x,
        &hamming_window,
        &[Caption("Hamming window"),Color("red")],
    );

    let blackman_window : Vec<f64> = dsp::compute_blackman_window(128);
    let blackman_window_x = gen_signal_vec(&blackman_window);

    fg.axes2d().lines(
        &blackman_window_x,
        &blackman_window,
        &[Caption("Blackman window"),Color("red")],
    );
    
    fg.show().unwrap();
}

fn draw_designed_filter_sample(){
    let mut fg = Figure::new();
	fg.set_multiplot_layout(2, 2)
		.set_title("Designed low-pass filter demo")
		.set_scale(0.8, 0.8)
		.set_offset(0.0, 0.0)
		.set_multiplot_fill_order(RowsFirst, Downwards);

    let filter_kernel_lpf = dsp::design_windowed_sinc_filter(10000.0, 48000.0, 28);
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

    let filter_kernel_hpf = dsp::design_windowed_sinc_filter_hpf(10000.0, 48000.0,28);
    //let mut lpf_for_inversion = impulse_response::DESIGNED_LPF_6KHZ.to_vec();
    //dsp::perform_spectral_inversion(&mut lpf_for_inversion);
    let filtred_result_hpf = dsp::convolution(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ, &filter_kernel_hpf);
    fg.axes2d().lines(
		&filtred_signal_vec,
		&filtred_result_hpf,
		&[Caption("Filtred signal HPF"),Color("red")],
	);


    let mut impulse_sample:Vec<f64> = (0..256).map(|_x|{0 as f64}).collect();
    impulse_sample[0] = 1.0;

    let impulse_response_lpf = dsp::convolution(&impulse_sample, &filter_kernel_lpf);
    let impulse_response_vec = gen_signal_vec(&impulse_response_lpf);
    fg.axes2d().lines(
		&impulse_response_vec,
		&impulse_response_lpf,
		&[Caption("Impulse response LPF"),Color("red")],
	);

    fg.show().unwrap();
}


fn draw_designed_bandpass_filter(){
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

fn main(){
    let mean = dsp::compute_signal_mean(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ);
    let signal_variance = dsp::compute_signal_variance(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ);
    let signal_deviation = dsp::compute_signal_devication(signal_variance);

    println!("Array mean is : {}", mean);
    println!("Signal variance is : {}", signal_variance);
    println!("Signal deviation is : {}", signal_deviation);

    //draw_convolution_sample();
    //draw_dft_sample();
    //draw_fft_over_ecg();
    //draw_rectangular_to_polar_sample();
    //draw_20khz_rex_imx_sample_with_complex_dft();
    //draw_fft_vs_dft();
    //draw_hamming_blackman_windows();
    draw_designed_filter_sample();
    //draw_designed_bandpass_filter();
}
