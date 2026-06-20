use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, SizedSample, Stream};
use tokio::sync::mpsc;

pub const REALTIME_SAMPLE_RATE: u32 = 24_000;
pub const CHANNELS_MONO: u16 = 1;

pub struct AudioIo {
    input_stream: Stream,
    output_stream: Stream,
    playback: PlaybackQueue,
}

impl AudioIo {
    pub fn start(input_tx: mpsc::UnboundedSender<Vec<i16>>) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let input_device = host
            .default_input_device()
            .ok_or(AudioError::NoInputDevice)?;
        let output_device = host
            .default_output_device()
            .ok_or(AudioError::NoOutputDevice)?;

        let input_config = input_device.default_input_config()?;
        let output_config = output_device.default_output_config()?;
        let input_stream = build_input_stream(&input_device, &input_config, input_tx)?;
        let playback = PlaybackQueue::default();
        let output_stream = build_output_stream(&output_device, &output_config, playback.clone())?;

        input_stream.play()?;
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

#[cfg(test)]
mod tests {
    use super::*;

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
