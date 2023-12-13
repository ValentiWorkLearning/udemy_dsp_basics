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

	let filter_kernel_highpass = dsp::design_bandbass_filter(1000.0,2000.0, 48000.0, 50);
	let player = player::PlayerImpl{};
	player.play_with_applied_filter(String::from("./assets/Jazz.wav"),&filter_kernel_highpass);
}
