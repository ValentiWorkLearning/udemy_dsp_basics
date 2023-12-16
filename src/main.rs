mod dsp;
mod drawing_samples;
mod player;
use crate::player::Player;

fn main(){

    //drawing_samples::draw_convolution_sample();
    //draw_dft_sample();
    //draw_fft_over_ecg();
    //draw_rectangular_to_polar_sample();
    //draw_20khz_rex_imx_sample_with_complex_dft();
    //draw_fft_vs_dft();
    //draw_hamming_blackman_windows();
    //draw_designed_filter_sample();
    //draw_designed_bandpass_filter();
	//draw_hpf_hpf_impulse_step_response();

	// let filter_kernel_highpass = dsp::design_windowed_sinc_filter(100.0,48000.0, 60);
	// let player = player::PlayerImpl{};
	// player.play_with_applied_filter(String::from("/Users/valentynkorniienko/Documents/Development/dsp_course/assets/patron.wav"),&filter_kernel_highpass);

	drawing_samples::draw_amplitute_modulation_sample();
}
