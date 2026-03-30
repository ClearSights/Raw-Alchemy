# Fix lens correction scaling: wrong scale direction and cross-platform inconsistency

## Summary

Two bugs in `lensfun_wrapper.py` caused the auto-scale step of lens correction to
produce incorrect — and in one case non-deterministic — results. Both bugs affected
the same `apply_lens_correction()` function and are fixed together in this branch.

---

## Bug 1 — Incorrect scaling when `auto_scale < 1.0` and wrong operation order

### Problem

The original code applied TCA correction **after** the auto-scale call, meaning the
scaling modifier was registered without knowing about TCA. More critically, the
`get_auto_scale()` result was misinterpreted:

```python
# BEFORE — wrong
if correct_distortion:
    modifier.enable_distortion_correction()

auto_scale = modifier.get_auto_scale()
if auto_scale < 1.0:
    modifier.enable_scaling(1.0 / auto_scale)   # ← inverted the value
else:
    modifier.enable_scaling(auto_scale)

if correct_tca:
    modifier.enable_tca_correction()             # ← registered too late
```

`lf_modifier_get_auto_scale()` already returns the correct scale factor to apply
directly (i.e. the value that crops the distortion-warped image to eliminate black
borders). The `< 1.0` branch was inverting it, which would grow the image instead of
cropping it and would leave black borders in the majority of barrel-distortion lenses
where `auto_scale < 1.0`. Additionally, TCA must be registered before `get_auto_scale`
so the scale calculation accounts for the subpixel spread.

### Fix

```python
# AFTER — correct
if correct_distortion:
    modifier.enable_distortion_correction()

if correct_tca:
    modifier.enable_tca_correction()             # ← registered before auto-scale

if correct_distortion:
    auto_scale = modifier.get_auto_scale()
    modifier.enable_scaling(auto_scale)          # ← apply value directly
```

---

## Bug 2 — Scale factor inconsistency between Windows and Linux/macOS

### Problem

The scale factor produced by `get_auto_scale()` differed between platforms for the
same image and lens — and the two values were exact reciprocals of each other.
For example, with a Sony lens on WSL Ubuntu 24.04 vs. Windows 11 (same Python 3.11,
same lensfun database, same branch):

| Platform | Reported scale | Matches Lightroom? |
|---|---|---|
| WSL / Linux | 0.9670 | ✅ Yes |
| Windows 11 | 1.0341 | ❌ No (`1 / 0.9670`) |

The root cause was a missing parameter in the ctypes binding. The actual C signature is:

```c
// lensfun.h
float lf_modifier_get_auto_scale(lfModifier *modifier, cbool reverse);
```

The `reverse` parameter, when `true`, causes lensfun to return `1.0 / scale` instead
of `scale` — the inverse transform, used when you want to pre-distort an image rather
than correct it.

The Python binding omitted this parameter:

```python
# BEFORE — incorrect argtypes (missing reverse)
_lensfun.lf_modifier_get_auto_scale.argtypes = [ctypes.POINTER(lfModifier)]

def get_auto_scale(self) -> float:
    return _lensfun.lf_modifier_get_auto_scale(self.modifier)
    # ^ reverse is not passed; its value comes from whatever happens to be
    #   in the second argument register (RSI on Linux, RDX on Windows)
```

Under the **System V AMD64 ABI** (Linux/macOS), RSI happened to hold `0` at the call
site → `reverse = false` → correct result.
Under the **Microsoft x64 ABI** (Windows), RDX happened to hold a non-zero value →
`reverse = true` → the result was inverted.

This is undefined behaviour: the correct value was produced on Linux only by accident.

### Fix

```python
# AFTER — correct argtypes with explicit reverse parameter
_lensfun.lf_modifier_get_auto_scale.argtypes = [
    ctypes.POINTER(lfModifier),
    ctypes.c_int,   # cbool reverse
]

def get_auto_scale(self, reverse: bool = False) -> float:
    return _lensfun.lf_modifier_get_auto_scale(self.modifier, int(reverse))
```

Passing `reverse=False` explicitly makes the result deterministic and identical across
all platforms.

---

## Files changed

- `src/raw_alchemy/lensfun_wrapper.py`

## Testing

Verified with `test_data/_DSC7298.ARW` (Sony) on WSL Ubuntu 24.04 and Windows 11
using the bundled lensfun database. Both platforms now produce `auto_scale ≈ 0.9670`,
consistent with the correction applied by Lightroom.
