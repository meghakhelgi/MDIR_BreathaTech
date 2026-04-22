# Demo-Day GUI

Standalone desktop app for demo-day visualization of the current-to-PPM fit used in the MDIR workflow.

## What It Does

- accepts a manual PalmSens current value
- lets you choose the current unit directly in the GUI: `pA`, `nA`, `uA`, `mA`, or `A`
- converts that value into a displayed `PPM` value using the active mapping settings
- plots the fit curve live
- highlights adjustable proxy PPM markers on the curve
- shows the entered current directly on the plot for live audience walkthroughs
- lets you compensate for low-sensitivity days with a sensitivity multiplier
- lets you override slope/intercept and clip bounds if you recalculated the fit
- keeps the advanced mapping and proxy-marker controls inside collapsible sections for a cleaner demo layout
- optionally previews and sends a JSON payload to a user-defined FastAPI endpoint

## Default Conversion Used

By default, the GUI mirrors the same constants used in MDIR:

```text
raw_current_uA = input_current converted into uA
corrected_current_uA = raw_current_uA * sensitivity_multiplier
peroxide_mM = max((corrected_current_uA - 0.163) / 0.0913, 0)
ppm = clip((1.545 * peroxide_mM) - 1.03, 0, 25)
```

Default settings:

- sensitivity multiplier = `1.0`
- current-to-peroxide slope = `0.0913`
- current-to-peroxide intercept = `0.163`
- peroxide-to-PPM slope = `1.545`
- peroxide-to-PPM intercept = `-1.03`
- PPM clip range = `0` to `25`

## Adapting To Lower Sensor Response

If the sensor is reading lower current values than expected on a given day, you have two clean options:

1. Quick compensation:
   Increase the `Sensitivity multiplier` above `1.0`.
   Example: `1.20` means the GUI treats the measured current as 20% stronger before applying the fit.

2. Full recalibration:
   If you have a new calibration line, edit the slope and intercept fields directly so the GUI reflects that updated fit.

Use the multiplier for temporary drift or demo compensation.
Use the slope/intercept overrides when you actually have a recalculated calibration.

## Run

From this folder:

```powershell
python .\app.py
```

or:

```powershell
.\run_demo_day_gui.ps1
```

The PowerShell launcher is the safer option if `python` is not on your `PATH`, because it also checks common local Python install locations.

## UI Notes

- `Adaptive mapping controls` and `Adjustable proxy PPM markers` start collapsed and can be expanded when you want to explain the fit in more detail.
- The backend payload area can be shown or hidden with the checkbox near the bottom of the window.

## Backend Payload Panel

The backend panel is optional and can be hidden.

It uses a JSON template with the `{{PPM}}` placeholder. Example:

```json
{
  "eco_ppm": {{PPM}}
}
```

When a valid current is entered, the app renders the final JSON preview and can POST it to the backend URL you provide.

## Important Note About FastAPI Payloads

This app intentionally does not force the MDIR or BreathaTech schema.

That is on purpose, because this GUI is centered on the current-to-PPM conversion and you said the other fields may be supplied elsewhere.

If your chosen FastAPI endpoint expects more than just `eco_ppm`, edit the template so it matches that endpoint before sending.

## Quick Verification

Run the standard-library tests:

```powershell
python .\test_conversion.py
```
