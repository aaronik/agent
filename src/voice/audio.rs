use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, SizedSample, Stream};
use tokio::sync::mpsc;

pub const REALTIME_SAMPLE_RATE: u32 = 24_000;
pub const CHANNELS_MONO: u16 = 1;

pub struct AudioIo {
    input_stream: InputAudioStream,
    output_stream: Stream,
    playback: PlaybackQueue,
}

enum InputAudioStream {
    Cpal(Stream),
    #[cfg(target_os = "macos")]
    VoiceProcessing(VoiceProcessingInputStream),
}

impl AudioIo {
    pub fn start(input_tx: mpsc::UnboundedSender<Vec<i16>>) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let output_device = host
            .default_output_device()
            .ok_or(AudioError::NoOutputDevice)?;

        let input_stream = build_input_audio_stream(input_tx)?;
        let output_config = output_device.default_output_config()?;
        let playback = PlaybackQueue::default();
        let output_stream = build_output_stream(&output_device, &output_config, playback.clone())?;

        match &input_stream {
            InputAudioStream::Cpal(stream) => stream.play()?,
            #[cfg(target_os = "macos")]
            InputAudioStream::VoiceProcessing(stream) => stream.keepalive(),
        }
        output_stream.play()?;

        Ok(Self {
            input_stream,
            output_stream,
            playback,
        })
    }

    pub fn playback(&self) -> PlaybackQueue {
        self.playback.clone()
    }

    pub fn clear_playback(&self) {
        self.playback.clear();
    }

    pub fn keepalive(&self) {
        let _ = &self.input_stream;
        let _ = &self.output_stream;
    }
}

#[derive(Clone, Default)]
pub struct PlaybackQueue {
    inner: Arc<Mutex<PlaybackState>>,
}

#[derive(Default)]
struct PlaybackState {
    queue: VecDeque<i16>,
    generation: u64,
    last_activity: Option<Instant>,
}

impl PlaybackQueue {
    pub fn push_pcm16(&self, samples: &[i16]) {
        if let Ok(mut state) = self.inner.lock() {
            state.queue.extend(samples);
            if !samples.is_empty() {
                state.last_activity = Some(Instant::now());
            }
        }
    }

    pub fn clear(&self) {
        if let Ok(mut state) = self.inner.lock() {
            state.queue.clear();
            state.generation = state.generation.wrapping_add(1);
            state.last_activity = None;
        }
    }

    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .map(|state| state.queue.len())
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_active_within(&self, hangover: Duration) -> bool {
        self.inner
            .lock()
            .map(|state| {
                !state.queue.is_empty()
                    || state
                        .last_activity
                        .is_some_and(|last_activity| last_activity.elapsed() <= hangover)
            })
            .unwrap_or(false)
    }

    fn generation(&self) -> u64 {
        self.inner.lock().map(|state| state.generation).unwrap_or(0)
    }

    fn pop(&self) -> Option<i16> {
        let mut state = self.inner.lock().ok()?;
        let sample = state.queue.pop_front();
        if sample.is_some() {
            state.last_activity = Some(Instant::now());
        }
        sample
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("no default microphone/input device found")]
    NoInputDevice,
    #[error("no default speaker/output device found")]
    NoOutputDevice,
    #[error("audio stream error: {0}")]
    Stream(String),
    #[error(transparent)]
    DefaultStreamConfig(#[from] cpal::DefaultStreamConfigError),
    #[error(transparent)]
    BuildStream(#[from] cpal::BuildStreamError),
    #[error(transparent)]
    PlayStream(#[from] cpal::PlayStreamError),
}

fn build_input_audio_stream(
    tx: mpsc::UnboundedSender<Vec<i16>>,
) -> Result<InputAudioStream, AudioError> {
    #[cfg(target_os = "macos")]
    if use_macos_voice_processing_input() {
        match VoiceProcessingInputStream::start(tx.clone()) {
            Ok(stream) => return Ok(InputAudioStream::VoiceProcessing(stream)),
            Err(err) => eprintln!(
                "macOS voice processing input unavailable ({err}); falling back to default input"
            ),
        }
    }

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or(AudioError::NoInputDevice)?;
    let input_config = input_device.default_input_config()?;
    build_input_stream(&input_device, &input_config, tx).map(InputAudioStream::Cpal)
}

#[cfg(target_os = "macos")]
fn use_macos_voice_processing_input() -> bool {
    std::env::var("AGENT_MACOS_VOICE_PROCESSING")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    tx: mpsc::UnboundedSender<Vec<i16>>,
) -> Result<Stream, AudioError> {
    match config.sample_format() {
        SampleFormat::F32 => build_typed_input_stream::<f32>(device, config, tx),
        SampleFormat::I16 => build_typed_input_stream::<i16>(device, config, tx),
        SampleFormat::U16 => build_typed_input_stream::<u16>(device, config, tx),
        other => Err(AudioError::Stream(format!(
            "unsupported input sample format: {other:?}"
        ))),
    }
}

fn build_typed_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    tx: mpsc::UnboundedSender<Vec<i16>>,
) -> Result<Stream, AudioError>
where
    T: Sample + SizedSample,
    T: cpal::FromSample<i16>,
    i16: cpal::FromSample<T>,
{
    let stream_config = config.config();
    let channels = stream_config.channels as usize;
    let input_rate = stream_config.sample_rate.0;
    let mut resampler = LinearResampler::new(input_rate, REALTIME_SAMPLE_RATE);
    let err_fn = |err| eprintln!("input audio stream error: {err}");

    Ok(device.build_input_stream(
        &stream_config,
        move |data: &[T], _| {
            let mono = data.chunks(channels).map(|frame| {
                frame
                    .first()
                    .copied()
                    .unwrap_or(T::EQUILIBRIUM)
                    .to_sample::<i16>()
            });
            let chunk = resampler.process(mono);
            if !chunk.is_empty() {
                let _ = tx.send(chunk);
            }
        },
        err_fn,
        None,
    )?)
}

fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    playback: PlaybackQueue,
) -> Result<Stream, AudioError> {
    match config.sample_format() {
        SampleFormat::F32 => build_typed_output_stream::<f32>(device, config, playback),
        SampleFormat::I16 => build_typed_output_stream::<i16>(device, config, playback),
        SampleFormat::U16 => build_typed_output_stream::<u16>(device, config, playback),
        other => Err(AudioError::Stream(format!(
            "unsupported output sample format: {other:?}"
        ))),
    }
}

fn build_typed_output_stream<T>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    playback: PlaybackQueue,
) -> Result<Stream, AudioError>
where
    T: Sample + SizedSample + cpal::FromSample<i16>,
{
    let stream_config = config.config();
    let channels = stream_config.channels as usize;
    let output_rate = stream_config.sample_rate.0;
    let mut resampler = LinearResampler::new(REALTIME_SAMPLE_RATE, output_rate);
    let mut output_buffer = VecDeque::new();
    let mut observed_generation = playback.generation();
    let err_fn = |err| eprintln!("output audio stream error: {err}");

    Ok(device.build_output_stream(
        &stream_config,
        move |data: &mut [T], _| {
            let current_generation = playback.generation();
            if current_generation != observed_generation {
                observed_generation = current_generation;
                output_buffer.clear();
            }

            for frame in data.chunks_mut(channels) {
                while output_buffer.is_empty() {
                    let Some(input_sample) = playback.pop() else {
                        break;
                    };
                    output_buffer.extend(resampler.process(std::iter::once(input_sample)));
                }
                let sample = output_buffer.pop_front().unwrap_or(0).to_sample::<T>();
                for channel in frame {
                    *channel = sample;
                }
            }
        },
        err_fn,
        None,
    )?)
}

#[derive(Clone, Debug)]
pub struct LinearResampler {
    input_rate: u32,
    output_rate: u32,
    input_position: u64,
    next_output_position: u64,
    last_sample: i16,
}

impl LinearResampler {
    pub fn new(input_rate: u32, output_rate: u32) -> Self {
        Self {
            input_rate,
            output_rate,
            input_position: 0,
            next_output_position: 0,
            last_sample: 0,
        }
    }

    pub fn process<I>(&mut self, samples: I) -> Vec<i16>
    where
        I: IntoIterator<Item = i16>,
    {
        if self.input_rate == self.output_rate {
            return samples.into_iter().collect();
        }

        let mut output = Vec::new();
        for sample in samples {
            while self
                .next_output_position
                .saturating_mul(u64::from(self.input_rate))
                <= self
                    .input_position
                    .saturating_mul(u64::from(self.output_rate))
            {
                output.push(self.last_sample);
                self.next_output_position += 1;
            }
            self.last_sample = sample;
            self.input_position += 1;
        }
        output
    }
}

#[cfg(target_os = "macos")]
struct VoiceProcessingInputStream {
    audio_unit: coreaudio::audio_unit::AudioUnit,
}

#[cfg(target_os = "macos")]
impl VoiceProcessingInputStream {
    fn keepalive(&self) {
        let _ = &self.audio_unit;
    }

    fn start(tx: mpsc::UnboundedSender<Vec<i16>>) -> Result<Self, AudioError> {
        use coreaudio::audio_unit::{AudioUnit, Element, IOType, Scope, StreamFormat};
        use coreaudio::sys;

        let mut audio_unit = AudioUnit::new(IOType::VoiceProcessingIO)
            .map_err(|err| AudioError::Stream(format!("VoiceProcessingIO init failed: {err}")))?;
        audio_unit
            .uninitialize()
            .map_err(|err| AudioError::Stream(format!("uninitialize voice input failed: {err}")))?;

        let enable_input = 1u32;
        audio_unit
            .set_property(
                sys::kAudioOutputUnitProperty_EnableIO,
                Scope::Input,
                Element::Input,
                Some(&enable_input),
            )
            .map_err(|err| AudioError::Stream(format!("enable voice input failed: {err}")))?;

        let disable_output = 0u32;
        audio_unit
            .set_property(
                sys::kAudioOutputUnitProperty_EnableIO,
                Scope::Output,
                Element::Output,
                Some(&disable_output),
            )
            .map_err(|err| AudioError::Stream(format!("disable voice output failed: {err}")))?;

        let input_asbd: sys::AudioStreamBasicDescription = audio_unit
            .get_property(
                sys::kAudioUnitProperty_StreamFormat,
                Scope::Output,
                Element::Input,
            )
            .map_err(|err| AudioError::Stream(format!("get voice input format failed: {err}")))?;
        let input_format = StreamFormat::from_asbd(input_asbd)
            .map_err(|err| AudioError::Stream(format!("parse voice input format failed: {err}")))?;
        set_voice_processing_callback(&mut audio_unit, input_format, tx)?;

        audio_unit
            .initialize()
            .map_err(|err| AudioError::Stream(format!("initialize voice input failed: {err}")))?;

        audio_unit
            .start()
            .map_err(|err| AudioError::Stream(format!("start voice input failed: {err}")))?;

        Ok(Self { audio_unit })
    }
}

#[cfg(target_os = "macos")]
fn set_voice_processing_callback(
    audio_unit: &mut coreaudio::audio_unit::AudioUnit,
    input_format: coreaudio::audio_unit::StreamFormat,
    tx: mpsc::UnboundedSender<Vec<i16>>,
) -> Result<(), AudioError> {
    use coreaudio::audio_unit::audio_format::LinearPcmFlags;
    use coreaudio::audio_unit::render_callback::Args;
    use coreaudio::audio_unit::render_callback::data::{Interleaved, NonInterleaved};
    use coreaudio::audio_unit::{SampleFormat, Scope};

    let input_rate = input_format.sample_rate as u32;
    if input_format.sample_format != SampleFormat::F32 {
        return Err(AudioError::Stream(format!(
            "unsupported voice input format: {:?}",
            input_format.sample_format
        )));
    }

    if input_format
        .flags
        .contains(LinearPcmFlags::IS_NON_INTERLEAVED)
    {
        let mut resampler = LinearResampler::new(input_rate, REALTIME_SAMPLE_RATE);
        audio_unit
            .set_input_callback(move |args: Args<NonInterleaved<f32>>| {
                if let Some(channel) = args.data.channels().next() {
                    let mono = channel.iter().map(|sample| float_sample_to_i16(*sample));
                    send_resampled_voice_input(&tx, &mut resampler, mono);
                }
                Ok(())
            })
            .map_err(|err| {
                AudioError::Stream(format!("set voice non-interleaved callback failed: {err}"))
            })?;
    } else {
        let channels = input_format.channels as usize;
        let mut resampler = LinearResampler::new(input_rate, REALTIME_SAMPLE_RATE);
        audio_unit
            .set_input_callback(move |args: Args<Interleaved<f32>>| {
                let mono = args
                    .data
                    .buffer
                    .chunks(channels)
                    .map(|frame| float_sample_to_i16(*frame.first().unwrap_or(&0.0)));
                send_resampled_voice_input(&tx, &mut resampler, mono);
                Ok(())
            })
            .map_err(|err| {
                AudioError::Stream(format!("set voice interleaved callback failed: {err}"))
            })?;
    }

    let _ = audio_unit.stream_format(Scope::Output);
    Ok(())
}

#[cfg(target_os = "macos")]
fn send_resampled_voice_input<I>(
    tx: &mpsc::UnboundedSender<Vec<i16>>,
    resampler: &mut LinearResampler,
    samples: I,
) where
    I: IntoIterator<Item = i16>,
{
    let out = resampler.process(samples);
    if !out.is_empty() {
        let _ = tx.send(out);
    }
}

#[cfg(target_os = "macos")]
impl Drop for VoiceProcessingInputStream {
    fn drop(&mut self) {
        let _ = self.audio_unit.stop();
    }
}

fn float_sample_to_i16(sample: f32) -> i16 {
    let sample = sample.clamp(-1.0, 1.0);
    if sample < 0.0 {
        (sample * 32768.0) as i16
    } else {
        (sample * 32767.0) as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_sample_conversion_clamps_to_pcm16() {
        assert_eq!(float_sample_to_i16(-2.0), -32768);
        assert_eq!(float_sample_to_i16(-1.0), -32768);
        assert_eq!(float_sample_to_i16(0.0), 0);
        assert_eq!(float_sample_to_i16(1.0), 32767);
        assert_eq!(float_sample_to_i16(2.0), 32767);
    }

    #[test]
    fn same_rate_resampler_passes_samples_through() {
        let mut resampler = LinearResampler::new(24_000, 24_000);
        assert_eq!(resampler.process([1, 2, 3]), vec![1, 2, 3]);
    }

    #[test]
    fn downsample_resampler_drops_samples_at_expected_ratio() {
        let mut resampler = LinearResampler::new(48_000, 24_000);
        assert_eq!(resampler.process([1, 2, 3, 4, 5]), vec![0, 2, 4]);
    }

    #[test]
    fn playback_queue_can_be_cleared_for_barge_in() {
        let queue = PlaybackQueue::default();
        queue.push_pcm16(&[1, 2, 3]);
        assert_eq!(queue.len(), 3);
        assert!(queue.is_active_within(Duration::from_secs(1)));
        queue.clear();
        assert_eq!(queue.len(), 0);
        assert!(!queue.is_active_within(Duration::from_secs(1)));
    }

    #[test]
    fn playback_clear_advances_generation_to_clear_device_buffer() {
        let queue = PlaybackQueue::default();
        let initial_generation = queue.generation();
        queue.clear();
        assert_ne!(queue.generation(), initial_generation);
    }
}
