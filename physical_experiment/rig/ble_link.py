"""
Copied 4 Nov 2026 (author) from the ICRA rig (code/ble_link.py) for the Robo-Kabaddi
robot runs, unchanged apart from this note; the original is not edited.

ble_link.py — drop-in replacement for fsr_listener.py, over Bluetooth LE.

WHY
---
The ESP32's built-in WiFi trips TG1WDT on the Smorphi board. BLE does not:
Paper 1 ran BLE with the radio to itself (WiFi disabled) and it worked. This
module keeps that known-good radio path and removes the external ESP8266.

API — identical to the four fsr_listener functions the experiment actually uses:
    start_listeners(robot_ids=None)
    is_connected(robot_id) -> bool
    wait_for_all_connections(robot_ids, timeout) -> bool
    send_command(robot_id, cmd_char, retries=1, retry_delay=0.0) -> bool

so robot_controller.py and smorphi_controller.py need NO changes. In the
experiment script, install it before importing them:

    import ble_link, sys
    sys.modules["fsr_listener"] = ble_link          # must precede the import
    from robot_controller import RobotController

DESIGN
------
bleak is async; the experiment loop is not. One asyncio loop runs in a daemon
thread. send_command() is called from the OpenCV loop at 5-10 Hz per robot and
must never block, so it does NOT await the write: it stores the character and a
per-robot writer task sends it. Commands COALESCE — if two arrive before the
radio drains, only the newest is sent. That is the correct behaviour for a
control loop (a stale velocity command is worthless) and it stops BLE
backpressure turning into a growing queue of futures.

SETUP
-----
    python3 -m pip install bleak

macOS will prompt for Bluetooth permission the first time; grant it to Terminal
(System Settings -> Privacy & Security -> Bluetooth), or every scan returns
nothing with no error.
"""
from __future__ import annotations

import asyncio
import threading
import time

try:
    from bleak import BleakClient, BleakScanner
except ImportError:                                             # pragma: no cover
    raise SystemExit("ble_link needs bleak:  python3 -m pip install bleak")

# ---------------------------------------------------------------- config ---
# Must match BLE_NAME in the firmware.
BLE_NAMES = {
    "raider_1":   "Smorphi_Raider1",
    "defender_1": "Smorphi_Defender1",
    "defender_2": "Smorphi_Defender2",
    "defender_3": "Smorphi_Defender3",
}

# Nordic UART Service — the RX characteristic is the one the robot listens on.
CHAR_RX = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

SCAN_TIMEOUT_S  = 12.0
WRITE_PERIOD_S  = 0.04     # 25 Hz ceiling per robot; plenty for a 10 Hz loop

# The firmware declares the RX characteristic as PROPERTY_WRITE only -- it does
# NOT advertise write-without-response. macOS can silently discard a
# write-without-response to such a characteristic: no exception, no movement.
# So write WITH response by default. Set False only if ble_diag.py confirms the
# characteristic advertises write-without-response and you need the latency.
WRITE_WITH_RESPONSE = True
RECONNECT_WAIT_S = 3.0

# ---------------------------------------------------------------- state ----
_connected: dict[str, bool] = {}
_pending:   dict[str, str | None] = {}
_last_sent: dict[str, str | None] = {}
_last_t:    dict[str, float] = {}
_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_started: set[str] = set()


# ---------------------------------------------------------------- worker ---
async def _robot_task(rid: str, name: str):
    """Connect to one robot, then write its pending command until cancelled."""
    while True:
        dev = None
        try:
            print(f"[BLE] scanning for {name} ({rid}) ...")
            dev = await BleakScanner.find_device_by_name(name, timeout=SCAN_TIMEOUT_S)
        except Exception as e:
            print(f"[BLE] {rid}: scan error {e}")

        if dev is None:
            print(f"[BLE] {rid}: '{name}' not found — retrying")
            await asyncio.sleep(RECONNECT_WAIT_S)
            continue

        try:
            async with BleakClient(dev) as client:
                with _lock:
                    _connected[rid] = True
                    _last_sent[rid] = None
                    _last_t[rid] = 0.0
                print(f"[BLE] {rid}: CONNECTED  ({dev.address})")

                while client.is_connected:
                    with _lock:
                        cmd = _pending.get(rid)
                        dup = (cmd is not None and cmd == _last_sent.get(rid)
                               and (time.monotonic() - _last_t.get(rid, 0.0)) < 0.5)
                    if cmd is not None and not dup:
                        try:
                            await client.write_gatt_char(
                                CHAR_RX, cmd.encode(),
                                response=WRITE_WITH_RESPONSE)
                            with _lock:
                                _last_sent[rid] = cmd
                                _last_t[rid] = time.monotonic()
                        except Exception as e:
                            print(f"[BLE] {rid}: write failed ({e})")
                            break
                    await asyncio.sleep(WRITE_PERIOD_S)
        except Exception as e:
            print(f"[BLE] {rid}: connection error {e}")
        finally:
            with _lock:
                _connected[rid] = False
            print(f"[BLE] {rid}: disconnected — reconnecting")
            await asyncio.sleep(RECONNECT_WAIT_S)


def _loop_thread():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


# ------------------------------------------------------------- public API --
def start_listeners(robot_ids=None):
    """Begin connecting to each robot. Returns immediately (non-blocking)."""
    global _thread
    if robot_ids is None:
        robot_ids = list(BLE_NAMES.keys())

    if _thread is None:
        _thread = threading.Thread(target=_loop_thread, daemon=True,
                                   name="ble-loop")
        _thread.start()
        while _loop is None:
            time.sleep(0.01)

    for rid in robot_ids:
        if rid in _started:
            continue
        name = BLE_NAMES.get(rid)
        if name is None:
            print(f"[BLE] no BLE name configured for {rid} — skipped")
            continue
        with _lock:
            _connected.setdefault(rid, False)
            _pending.setdefault(rid, None)
            _last_sent.setdefault(rid, None)
        asyncio.run_coroutine_threadsafe(_robot_task(rid, name), _loop)
        _started.add(rid)
        print(f"[BLE] task started for {rid} -> {name}")


def is_connected(robot_id) -> bool:
    with _lock:
        return bool(_connected.get(robot_id, False))


def wait_for_all_connections(robot_ids=None, timeout=90.0) -> bool:
    if robot_ids is None:
        robot_ids = list(BLE_NAMES.keys())
    t0 = time.time()
    last = 0.0
    while time.time() - t0 < timeout:
        missing = [r for r in robot_ids if not is_connected(r)]
        if not missing:
            return True
        if time.time() - last > 5.0:
            last = time.time()
            print(f"[BLE] waiting for: {', '.join(missing)}")
        time.sleep(0.25)
    return False


def send_command(robot_id, cmd_char, retries=1, retry_delay=0.0) -> bool:
    """Queue one ASCII command. Non-blocking; newest command wins.

    `retries`/`retry_delay` are accepted for API compatibility and ignored:
    they existed to survive ESP8266 AT half-duplex. BLE writes are ordered and
    acknowledged at the link layer, and re-sending would only add latency.
    """
    if not cmd_char or len(cmd_char) != 1:
        return False
    with _lock:
        if not _connected.get(robot_id, False):
            return False
        _pending[robot_id] = cmd_char
    return True


def stop_all():
    """Send the stop character to every connected robot."""
    for rid in list(_connected):
        send_command(rid, "}")
    time.sleep(0.3)          # let the writer tasks drain before any shutdown


def shutdown():
    """Stop cleanly. Without this, Python tears down the daemon loop mid-await
    and prints 'Task was destroyed but it is pending' / 'coroutine ignored
    GeneratorExit' — harmless, but noise that can hide a real error."""
    stop_all()
    if _loop is not None and _loop.is_running():
        for t in asyncio.all_tasks(_loop):
            _loop.call_soon_threadsafe(t.cancel)
        time.sleep(0.2)
        _loop.call_soon_threadsafe(_loop.stop)
        time.sleep(0.2)


# ------------------------------------------------------------- self-test ---
if __name__ == "__main__":
    ids = list(BLE_NAMES.keys())
    start_listeners(ids)
    if wait_for_all_connections(ids, timeout=60.0):
        print("\n[BLE] all robots connected")
    else:
        print("\n[BLE] some robots missing:")
        for r in ids:
            print(f"   {r}: {'CONNECTED' if is_connected(r) else 'NOT CONNECTED'}")
    try:
        while True:
            time.sleep(2.0)
            print("  " + "  ".join(
                f"{r}={'ok' if is_connected(r) else '--'}" for r in ids))
    except KeyboardInterrupt:
        stop_all()
        print("\nstopped")
