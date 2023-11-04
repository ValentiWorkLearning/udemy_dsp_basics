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
    

    let convolution_result = dsp::convolution(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ, &impulse_response::IMPULSE_RESPONSE);
    let convolution_vec = gen_signal_vec(&convolution_result);
    fg.axes2d().lines(
		&convolution_vec,
		&convolution_result,
		&[Caption("Convolution"),Color("blue")],
	);


    let running_sum_result = dsp::running_sum(&waveforms::INPUT_SIGNAL_32_1K_HZ_15K_HZ);
    let running_sum_vec = gen_signal_vec(&running_sum_result);

    fg.axes2d().lines(
		&running_sum_vec,
		&running_sum_result,
		&[Caption("Running sum"),Color("blue")],
	);

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
    draw_rectangular_to_polar_sample();
}
