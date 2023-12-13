use std::env;
use std::fs::File;
use std::path::Path;

use crate::dsp;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::sample;

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, Sample, SizedSample,
};

pub trait Player {
    fn play_with_applied_filter(&self, file_path: String, filter: &Vec<f32>);
}

pub struct PlayerImpl {}

fn write_data<T>(output: &mut [T], channels: usize, next_sample: &mut dyn FnMut() -> f32)
where
    T: Sample + FromSample<f32>,
{
    for frame in output.chunks_mut(channels) {
        let value: T = T::from_sample(next_sample());
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}
impl Player for PlayerImpl {
    fn play_with_applied_filter(&self, file_path: String, filter: &Vec<f32>) {
        // Create a media source. Note that the MediaSource trait is automatically implemented for File,
        // among other types.
        let file = Box::new(File::open(Path::new(file_path.as_str())).unwrap());

        // Create the media source stream using the boxed media source from above.
        let mss = MediaSourceStream::new(file, Default::default());

        // Create a hint to help the format registry guess what format reader is appropriate. In this
        // example we'll leave it empty.
        let hint = Hint::new();

        // Use the default options when reading and decoding.
        let format_opts: FormatOptions = Default::default();
        let metadata_opts: MetadataOptions = Default::default();
        let decoder_opts: DecoderOptions = Default::default();

        // Probe the media source stream for a format.
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .unwrap();

        // Get the format reader yielded by the probe operation.
        let mut format = probed.format;

        // Get the default track.
        let track = format.default_track().unwrap();

        // Create a decoder for the track.
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .unwrap();

        // Store the track identifier, we'll use it to filter packets.
        let track_id = track.id;

        let mut sample_count = 0;
        let mut sample_buf = None;


        let mut audio_stream_buffered : Vec<f32> = Vec::new();

        loop {
            // Get the next packet from the format reader.
            let packet_expected = format.next_packet();
            if packet_expected.is_err() {
                break;
            }
            let packet = packet_expected.unwrap();
            // If the packet does not belong to the selected track, skip it.
            if packet.track_id() != track_id {
                continue;
            }

            // Decode the packet into audio samples, ignoring any decode errors.
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    // The decoded audio samples may now be accessed via the audio buffer if per-channel
                    // slices of samples in their native decoded format is desired. Use-cases where
                    // the samples need to be accessed in an interleaved order or converted into
                    // another sample format, or a byte buffer is required, are covered by copying the
                    // audio buffer into a sample buffer or raw sample buffer, respectively. In the
                    // example below, we will copy the audio buffer into a sample buffer in an
                    // interleaved order while also converting to a f32 sample format.

                    // If this is the *first* decoded packet, create a sample buffer matching the
                    // decoded audio buffer format.
                    if sample_buf.is_none() {
                        // Get the audio buffer specification.
                        let spec = *audio_buf.spec();

                        // Get the capacity of the decoded buffer. Note: This is capacity, not length!
                        let duration = audio_buf.capacity() as u64;

                        // Create the f32 sample buffer.
                        sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                    }

                    // Copy the decoded audio buffer into the sample buffer in an interleaved format.
                    if let Some(buf) = &mut sample_buf {
                        buf.copy_interleaved_ref(audio_buf);

                        // The samples may now be access via the `samples()` function.
                        sample_count += buf.samples().len();
                        audio_stream_buffered.extend_from_slice(buf.samples());

                        print!("\rDecoded {} samples", sample_count);
                    }
                }
                Err(Error::DecodeError(_)) => (),
                Err(_) => break,
            }
        }

        let mut samples_extracted = sample_buf.unwrap();
        let audio_samples = samples_extracted.samples_mut();
        let result_samples = dsp::convolution(&audio_stream_buffered, &filter);

        self.play_on_default_streamer(&result_samples);
    }
}

impl PlayerImpl {
    fn play_on_default_streamer(&self, samples: &Vec<f32>) {
        let host = cpal::default_host();

        let device = host.default_output_device().unwrap();

        println!("Output device: {}", device.name().unwrap());

        let config = device.default_output_config().unwrap();
        println!("Default output config: {:?}", config);

        let channels = 2;
        let samples = samples.clone();
        let mut sample_clock = 0_usize;
        let mut next_value = move || {
            if sample_clock == samples.len() {
                sample_clock = 0;
            }
            let sampled = samples[sample_clock];
            sample_clock = sample_clock + 1;
            sampled
        };

        let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

        let stream = device
            .build_output_stream(
                &config.into(),
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    write_data(data, channels, &mut next_value)
                },
                err_fn,
                None,
            )
            .unwrap();
        stream.play();

        std::thread::sleep(std::time::Duration::from_millis(20000));
    }
}
