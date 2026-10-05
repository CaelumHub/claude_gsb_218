"""Sanity tests for silence detection / trimming (run directly, not pytest)."""
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend import audio_io, silence

SR = 22050
random.seed(7)


def tone(dur, amp=0.3, freq=300.0, noise=0.0):
    n = int(SR * dur)
    out = []
    for i in range(n):
        v = amp * math.sin(2 * math.pi * freq * i / SR)
        if noise:
            v += noise * random.uniform(-1, 1)
        out.append(v)
    return out


def silence_seg(dur, noise=0.0):
    n = int(SR * dur)
    if noise:
        return [noise * random.uniform(-1, 1) for _ in range(n)]
    return [0.0] * n


def save(samples, path):
    audio_io.save(path, audio_io.AudioData([samples], SR))


def approx(a, b, tol=0.12):
    return abs(a - b) <= tol


def main():
    tmp = tempfile.mkdtemp()

    # Case 1: 1s head silence + 2s tone + 1.2s internal silence + 2s tone + 0.8 tail
    src = os.path.join(tmp, "basic.wav")
    sig = silence_seg(1.0) + tone(2.0) + silence_seg(1.2) + tone(2.0) + silence_seg(0.8)
    save(sig, src)
    rep = silence.detect_silence(src)
    positions = [r["position"] for r in rep["regions"]]
    assert positions == ["leading", "internal", "trailing"], positions
    assert approx(rep["regions"][0]["end"], 1.0, 0.08), rep["regions"][0]
    assert approx(rep["regions"][1]["duration"], 1.2, 0.1), rep["regions"][1]
    assert approx(rep["stats"]["ratio"], 3.0 / 7.0, 0.03), rep["stats"]["ratio"]
    assert not rep["all_silence"] and not rep["no_silence"]
    print("case1 detect OK:", [(r["position"], round(r["start"], 2), round(r["end"], 2)) for r in rep["regions"]])

    # Trim edges + compress middle to 0.3s; expected duration derived from the
    # reported regions (boundaries move slightly with envelope smoothing).
    dst = os.path.join(tmp, "basic_trim.wav")
    rep2, sm = silence.trim_silence(src, dst, middle="keep", keep_seconds=0.3)
    rmap = {r["position"]: r for r in rep2["regions"]}
    expected = 7.0
    expected -= max(0.0, rmap["leading"]["end"] - 0.02)
    expected -= max(0.0, rmap["trailing"]["duration"] - 0.02)
    expected -= max(0.0, rmap["internal"]["duration"] - 0.3)
    assert approx(sm["result_duration"], expected, 0.06), (sm, expected)
    assert 2.4 < sm["removed_seconds"] < 2.7, sm
    assert not sm["all_silence"]
    with audio_io.WavReader(dst) as r:
        assert r.nframes > 0
        # Edge padding keeps 20 ms before the detected boundary; the envelope
        # follower's ~20 ms onset lag means the output effectively starts at
        # the tone onset (no silence, no clipped attack — inaudible join).
        head = r.read_excerpt(0, SR // 10).samples[0]
        assert max(abs(v) for v in head[:2000]) > 0.15
        # and the onset ramps in (a hard cut would start at a zero crossing
        # followed immediately by full amplitude — either is safe, but verify
        # the very first frames aren't clipped mid-cycle at high amplitude).
        assert abs(head[0]) < 0.5
    print("case1 trim OK: %.2f -> %.2f" % (sm["original_duration"], sm["result_duration"]))

    # Remove middle entirely
    dst2 = os.path.join(tmp, "basic_del.wav")
    _, sm2 = silence.trim_silence(src, dst2, middle="remove")
    assert 4.0 <= sm2["result_duration"] <= 4.2, sm2
    print("case1 delete middle OK: %.2f" % sm2["result_duration"])

    # Leave everything off -> unchanged duration
    dst3 = os.path.join(tmp, "basic_nop.wav")
    _, sm3 = silence.trim_silence(src, dst3, middle="leave", trim_edges=False)
    assert approx(sm3["result_duration"], 7.0, 0.02), sm3
    print("case1 no-op OK")

    # Case 2: floor noise (-45dB-ish) throughout; tone gaps still detected via adaptive
    src2 = os.path.join(tmp, "noisy.wav")
    floor = 10 ** (-45 / 20)
    gap = silence_seg(1.5, noise=floor * 1.4)
    sig2 = tone(1.5, amp=0.25, noise=floor) + gap + tone(1.5, amp=0.25, noise=floor)
    save(sig2, src2)
    rep_n = silence.detect_silence(src2)
    assert len(rep_n["regions"]) == 1 and rep_n["regions"][0]["position"] == "internal", \
        [(r["position"], round(r["mean_db"], 1)) for r in rep_n["regions"]]
    assert rep_n["threshold"]["threshold_db"] > rep_n["threshold"]["noise_floor_db"]
    assert rep_n["threshold"]["threshold_db"] < rep_n["threshold"]["active_db"] - 3
    print("case2 noisy floor OK: noise=%.1f thr=%.1f active=%.1f" % (
        rep_n["threshold"]["noise_floor_db"], rep_n["threshold"]["threshold_db"], rep_n["threshold"]["active_db"]))

    # Case 3: short speech pauses (0.12s) ignored at default min_silence=0.3
    src3 = os.path.join(tmp, "pauses.wav")
    sig3 = tone(1.0) + silence_seg(0.12) + tone(1.0) + silence_seg(0.1) + tone(1.0)
    save(sig3, src3)
    rep_p = silence.detect_silence(src3)
    assert rep_p["regions"] == [], rep_p["regions"]
    assert rep_p["no_silence"]
    # lower min_silence -> both detected
    rep_p2 = silence.detect_silence(src3, min_silence=0.05)
    assert len(rep_p2["regions"]) == 2, len(rep_p2["regions"])
    print("case3 short pauses OK")

    # Case 4: quiet content must NOT be flagged (-30dBFS steady tone between loud parts)
    src4 = os.path.join(tmp, "quiet.wav")
    sig4 = tone(1.5, amp=0.4) + tone(2.0, amp=0.03) + tone(1.5, amp=0.4)
    save(sig4, src4)
    rep_q = silence.detect_silence(src4)
    assert rep_q["regions"] == [], [(r["start"], r["end"], r["mean_db"]) for r in rep_q["regions"]]
    print("case4 quiet content preserved OK")

    # Case 5: isolated click inside a long silence doesn't split it (min_sound bridge)
    src5 = os.path.join(tmp, "click.wav")
    click = [0.0] * (SR // 20)
    sig5 = tone(1.0) + silence_seg(0.6) + click + silence_seg(0.6) + tone(1.0)
    save(sig5, src5)
    rep_c = silence.detect_silence(src5)
    internals = [r for r in rep_c["regions"] if r["position"] == "internal"]
    assert len(internals) == 1, [(r["start"], r["end"]) for r in internals]
    assert approx(internals[0]["duration"], 1.25, 0.08), internals[0]
    print("case5 click bridge OK")

    # Case 6: almost no silence anywhere
    src6 = os.path.join(tmp, "nonsilent.wav")
    save(tone(4.0), src6)
    rep_6 = silence.detect_silence(src6)
    assert rep_6["no_silence"] and rep_6["regions"] == [], rep_6["regions"]
    dst6 = os.path.join(tmp, "nonsilent_out.wav")
    _, sm6 = silence.trim_silence(src6, dst6)
    assert approx(sm6["result_duration"], 4.0, 0.02), sm6
    print("case6 no-silence extreme OK")

    # Case 7: almost all silence
    src7 = os.path.join(tmp, "allsilent.wav")
    sig7 = silence_seg(3.0) + tone(0.15, amp=0.2) + silence_seg(3.0)
    save(sig7, src7)
    rep_7 = silence.detect_silence(src7)
    assert rep_7["all_silence"] is True, rep_7["stats"]
    dst7 = os.path.join(tmp, "allsilent_out.wav")
    _, sm7 = silence.trim_silence(src7, dst7)
    assert sm7["mostly_silence"]
    # the 0.15 s of content survives; the 6 s of edges are gone
    assert approx(sm7["result_duration"], 0.19, 0.06), sm7["result_duration"]
    with audio_io.WavReader(dst7) as r:
        assert r.nframes > 0
    print("case7 all-silence extreme OK: out=%.2f" % sm7["result_duration"])

    # Case 8: manual threshold respected
    rep_8 = silence.detect_silence(src4, threshold_db=-20.0)  # -30dB quiet part now "silent"
    assert len(rep_8["regions"]) == 1, [(r["start"], r["end"], r["mean_db"]) for r in rep_8["regions"]]
    rep_8b = silence.detect_silence(src4, threshold_db=-60.0)
    assert rep_8b["regions"] == []
    print("case8 manual threshold OK")

    # Case 9: stereo path + crossfade sanity (no clipping, no pops)
    stereo = os.path.join(tmp, "stereo.wav")
    audio_io.save(stereo, audio_io.AudioData([sig, sig], SR))
    dst9 = os.path.join(tmp, "stereo_out.wav")
    _, sm9 = silence.trim_silence(stereo, dst9, middle="remove")
    with audio_io.WavReader(dst9) as r:
        assert r.channels == 2
        peak = 0.0
        for ch in r.iter_chunks():
            for c in ch:
                for v in c:
                    if abs(v) > peak:
                        peak = abs(v)
        assert peak <= 1.0, peak
    print("case9 stereo/clip OK: %.2fs peak=%.2f" % (sm9["result_duration"], peak))

    # Case 10: selection restricts which regions are acted on
    dst10 = os.path.join(tmp, "sel.wav")
    idx_internal = [r["index"] for r in rep2["regions"] if r["position"] == "internal"]
    _, sm10 = silence.trim_silence(src, dst10, middle="remove", trim_edges=True,
                                   selection=idx_internal)
    # only the internal gap removed (~1.2s), edges kept
    assert approx(sm10["result_duration"], 7.0 - 1.2, 0.1), sm10
    print("case10 selection-only OK: %.2f" % sm10["result_duration"])

    print("\nALL SILENCE TESTS PASSED")


if __name__ == "__main__":
    main()
