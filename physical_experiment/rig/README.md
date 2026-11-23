# Robot rig for the 2.6 x 4.0 m court (Bluetooth robots, fisheye camera)

Set up 4 Nov 2026 (author). Copies, originals unchanged:
`camera_model.py` and `calibrate_rig.py` from the Paper 2 rig (court changed to
2.6 x 4.0 m), `ble_link.py` from the ICRA rig. The older Wi-Fi code and
4-corner calibration in `physical_experiment/` are left as they were (April runs).

Run everything from the repository root with the `kabaddi` conda env.

## 1. Camera calibration (after any camera move)

    python3 physical_experiment/rig/calibrate_rig.py --camera-height <m> --marker-height <m>

Click the 4 court corners (top-left = defenders' end, raider's left; then
top-right, bottom-right, bottom-left), then 5–8 points along each line: defenders'
end line, raider's end line, left side line, right side line, mid-line. SPACE
moves to the next step. The fit prints how far the points are from their lines
(should be under 3 cm); the top-down check window should show straight court
lines on the window edges. SPACE saves `physical_experiment/rig/camera_calib.json`.

## 1b. Position check

    python3 physical_experiment/rig/check_positions.py --marker 3

Place the robot centred over each of the 8 spots (measured with a tape; x from
the side line on the raider's left, y from the raider's end line), SPACE at
each. Writes `position_check_<time>.csv`; a few cm is good.

## 2. B2/B3 speed trials

    python3 physical_experiment/rig/b3_drive.py --robot raider_1

Keys in the window: `f` forward, `b` backward, `r` diagonal forward-right,
`l` diagonal forward-left (all 100%), SPACE stop, `q` quit. Each key is one
trial; the robot stops itself after 2 m, near a line (0.30 m), after 10 s, or if
its marker is lost. Plan per robot: 5 x `f`, 5 x `r`, 5 x `l`. Then:

    python3 physical_experiment/phase_b/analyse_motion.py \
        --log physical_experiment/phase_b/data/session_*/trials.csv \
        --sheet physical_experiment/phase_b/trials_sheet.csv

## Checks without robots

    python3 physical_experiment/rig/test_camera_model.py        # 9 checks
    python3 physical_experiment/rig/b3_drive.py --fake --outdir <scratch folder>

## 3. Real raids with the trained policies (raid_runner.py)

First a direction check (all 4 robots switched on; one at a time in the
middle of the court, SPACE): each robot drives 0.6 s forward, right,
diagonal forward-right, diagonal forward-left and left, and the camera says
whether it went the expected way.

    python3 physical_experiment/rig/raid_runner.py --check

Then raids (raid i starts from the same places as simulated raid 50000 + i):

    python3 physical_experiment/rig/raid_runner.py --raider learned --defence learned --n-def 3 --raids 3
    python3 physical_experiment/rig/raid_runner.py --raider learned --defence scripted --gate v8 --n-def 3 --raids 3
    python3 physical_experiment/rig/raid_runner.py --raider scripted --defence learned --n-def 3 --raids 3

Before each raid the runner drives each robot to its start place (circles in
the camera window, within 10 cm), one at a time; then SPACE to start
(`--manual-place` to put them by hand). SPACE during a raid stops all robots. Results:
`physical_experiment/match_data/session_<time>/raids.csv` and `steps.csv`.
Without robots: add `--fake --outdir <scratch folder>`.
