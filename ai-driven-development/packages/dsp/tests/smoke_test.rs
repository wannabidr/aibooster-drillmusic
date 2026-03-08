use ai_dj_dsp;

#[test]
fn smoke_test_dsp_package_loads() {
    let version = ai_dj_dsp::version();
    assert!(!version.is_empty(), "DSP package should export a version");
}
