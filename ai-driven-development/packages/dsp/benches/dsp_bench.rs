use criterion::{criterion_group, criterion_main, Criterion};
use ai_dj_dsp::domain::{AudioBuffer, BeatGrid, CrossfadeCurve, CrossfadeStyle, TransitionParams};
use ai_dj_dsp::engine::CrossfadeEngine;
use ai_dj_dsp::{io, ffi};

fn make_track(duration_secs: usize, sample_rate: u32, channels: u16) -> AudioBuffer {
    let num_frames = sample_rate as usize * duration_secs;
    let samples: Vec<f32> = (0..num_frames * channels as usize)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
        })
        .collect();
    AudioBuffer::new(samples, sample_rate, channels).unwrap()
}

fn bench_crossfade_30s_mono(c: &mut Criterion) {
    let sr = 44100u32;
    let track_a = make_track(60, sr, 1);
    let track_b = make_track(60, sr, 1);
    let grid_a = BeatGrid::from_bpm(128.0, 0, sr, 60 * sr as u64).unwrap();
    let grid_b = BeatGrid::from_bpm(128.0, 0, sr, 60 * sr as u64).unwrap();
    let params = TransitionParams::default_32_beat(0, 0);

    c.bench_function("crossfade_30s_mono_128bpm", |b| {
        b.iter(|| {
            CrossfadeEngine::render(&track_a, &track_b, &grid_a, &grid_b, &params, 16).unwrap()
        })
    });
}

fn bench_crossfade_30s_stereo(c: &mut Criterion) {
    let sr = 44100u32;
    let track_a = make_track(60, sr, 2);
    let track_b = make_track(60, sr, 2);
    let grid_a = BeatGrid::from_bpm(128.0, 0, sr, 60 * sr as u64).unwrap();
    let grid_b = BeatGrid::from_bpm(128.0, 0, sr, 60 * sr as u64).unwrap();
    let params = TransitionParams::default_32_beat(0, 0);

    c.bench_function("crossfade_30s_stereo_128bpm", |b| {
        b.iter(|| {
            CrossfadeEngine::render(&track_a, &track_b, &grid_a, &grid_b, &params, 16).unwrap()
        })
    });
}

fn bench_crossfade_equal_power(c: &mut Criterion) {
    let sr = 44100u32;
    let track_a = make_track(60, sr, 2);
    let track_b = make_track(60, sr, 2);
    let grid_a = BeatGrid::from_bpm(128.0, 0, sr, 60 * sr as u64).unwrap();
    let grid_b = BeatGrid::from_bpm(128.0, 0, sr, 60 * sr as u64).unwrap();
    let params = TransitionParams::new(
        32,
        CrossfadeStyle::VolumeFade(CrossfadeCurve::EqualPower),
        0,
        0,
    ).unwrap();

    c.bench_function("crossfade_30s_stereo_equal_power", |b| {
        b.iter(|| {
            CrossfadeEngine::render(&track_a, &track_b, &grid_a, &grid_b, &params, 16).unwrap()
        })
    });
}

fn bench_full_pipeline_load_render_save(c: &mut Criterion) {
    let dir = std::env::temp_dir().join("ai_dj_bench");
    std::fs::create_dir_all(&dir).unwrap();

    let path_a = dir.join("bench_track_a.wav");
    let path_b = dir.join("bench_track_b.wav");

    // Create 120s stereo test WAV files
    let track_a = make_track(120, 44100, 2);
    let track_b = make_track(120, 44100, 2);
    io::save_wav(&track_a, &path_a).unwrap();
    io::save_wav(&track_b, &path_b).unwrap();

    c.bench_function("full_pipeline_load_render_save_stereo", |b| {
        b.iter(|| {
            ffi::render_crossfade(
                path_a.to_str().unwrap(),
                path_b.to_str().unwrap(),
                128.0,
                128.0,
                32,
                "equal_power",
            ).unwrap()
        })
    });

    // Cleanup
    std::fs::remove_file(&path_a).ok();
    std::fs::remove_file(&path_b).ok();
}

criterion_group!(
    benches,
    bench_crossfade_30s_mono,
    bench_crossfade_30s_stereo,
    bench_crossfade_equal_power,
    bench_full_pipeline_load_render_save
);
criterion_main!(benches);
