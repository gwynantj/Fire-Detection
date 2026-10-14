# Fire Watch: fire/smoke warnings for Frigate cameras

A small service that sits next to [Frigate](https://frigate.video), runs a fire/smoke detector on each camera's latest
frame and raises alarms in Home Assistant over MQTT.

```
Frigate /api/<camera>/latest.jpg ──► YOLOv8n fire/smoke (ONNX, CPU) ──► temporal persistence ──► MQTT ──► Home Assistant ──► critical push
```

- **Detector:** [`rabahdev/fire-smoke-yolov8n`](https://huggingface.co/rabahdev/fire-smoke-yolov8n), YOLOv8n trained on
  the D-Fire dataset (mAP50 0.754 on the D-Fire test split). The original YOLOv5 weights of this repository are no
  longer downloadable. The Dockerfile exports the model to ONNX at build time, so the runtime image needs no torch.
  Inference takes ~40 ms per frame on an i5-8xxx CPU.
- **Temporal analysis:** the temporal persistence technique (TPT) from the main project, recommended for indoor scenes.
  An alarm is raised when more than `PERSISTENCE_THRESH` of the last `WINDOW_SIZE` frames contain a detection, and
  cleared when that fraction drops below `CLEAR_THRESH`. With the defaults (1 fps, 20 frames, 0.5) a fire must be
  visible in more than 10 of the last 20 seconds.
- **Flame check:** fire detections below `FIRE_SURE_CONF` only count if the box looks like a flame: its content
  changes between frames more than the rest of the frame does (flames flicker) or it contains bright, light-emitting
  pixels. This removes static red/orange objects, which the model can score up to ~0.55, without raising `FIRE_CONF`
  and losing weak real fires. On test footage, real flames measured flicker 16-38 and bright 0.06-0.31, a static red
  bowl flicker 3-6 (the same as whole-frame sensor noise, 4-9) and bright 0.
- **Lighting changes:** switching the room lights flips the camera between IR (grayscale) and colour and changes the
  whole frame at once, which made static objects look like they flicker. A mode flip or a whole-frame change of at
  least `LIGHT_CHANGE` is logged as `lighting change`, and for `LIGHT_SETTLE` seconds afterwards unsure fire
  detections are ignored (`ignored static fire ... (lighting change)`). Fire at or above `FIRE_SURE_CONF` and smoke
  are unaffected.
- **Frames:** pulled from Frigate's HTTP API, so the camera stream is not decoded twice. If the frame stops changing for
  `STALE_AFTER` seconds, the camera is marked unavailable.

## MQTT / Home Assistant

The service publishes Home Assistant discovery messages and creates one device ("Fire Watch") with, per camera:

| Entity | Description |
|---|---|
| `binary_sensor.firewatch_<camera>_fire` | Confirmed fire (device class `safety`) |
| `binary_sensor.firewatch_<camera>_smoke` | Confirmed smoke (device class `smoke`) |
| `binary_sensor.firewatch_<camera>_lights` | Room lights, inferred from the camera being in colour (on) or IR (off) mode |
| `camera.firewatch_<camera>_snapshot` | Annotated frame from the latest alarm |

Each binary sensor has `persistence` and `best_score` attributes. Both sensors become `unavailable` when the service is
down (MQTT last will) or the camera feed is stale, so a broken detector is visible rather than silently "off".

[`homeassistant/automations.yaml`](homeassistant/automations.yaml) contains the alert automations. Append it to
Home Assistant's `automations.yaml` and reload automations:

- **Alarm:** critical iOS notification (bypasses silent/focus) with the snapshot, repeated every 2 minutes while active.
- **Cleared:** notification when the alarm turns off.
- **Offline:** notification if detection has been unavailable for 5 minutes, plus a follow-up when it recovers.

## Reviewing detections

Every detection episode, including brief ones that never become an alarm, is logged as a Frigate event on the
camera, labelled `fire` or `smoke`. The sub label is `firewatch` for raw detections and `firewatch alarm` once
the temporal analysis confirms it. Each event has a snapshot with the detection boxes and a clip from the camera's
recordings. Browse them in Frigate (Review / Explore, filter by label) or in Home Assistant with the
Advanced Camera Card (see `homeassistant/dashboard.yaml`).

## Deploy

```
cd firewatch
docker compose up -d --build
docker logs -f firewatch
```

The compose file uses host networking to reach Frigate (`:5000`) and Mosquitto (`:1883`). Configuration is via
environment variables:

| Variable | Default | Description |
|---|---|---|
| `FRIGATE_URL` | `http://127.0.0.1:5000` | Frigate API (unauthenticated port) |
| `CAMERAS` | `chp` | Comma-separated Frigate camera names |
| `INTERVAL` | `1.0` | Seconds between frames per camera |
| `SMOKE_CONF` / `FIRE_CONF` | `0.45` / `0.40` | Per-class confidence thresholds |
| `FIRE_SURE_CONF` | `0.6` | Fire detections at or above this skip the flame check |
| `FLICKER_MIN` | `6` | Mean grayscale change (0-255) in the box between frames, above the whole frame's change, that counts as flicker |
| `BRIGHT_MIN` | `0.03` | Fraction of box pixels with brightness >= 220 that counts as emitted light |
| `IR_SAT_MAX` | `2` | Mean saturation (0-255) below which a frame is IR night mode (colour frames are ~25) |
| `LIGHT_CHANGE` | `20` | Whole-frame mean grayscale change that counts as a lighting change (noise is 4-9) |
| `LIGHT_SETTLE` | `5` | Seconds after a lighting change during which unsure fire detections are ignored |
| `MIN_AREA` | `0.0005` | Ignore boxes smaller than this fraction of the frame |
| `WINDOW_SIZE` | `20` | Persistence window (frames) |
| `PERSISTENCE_THRESH` | `0.5` | Fraction of the window needed to raise an alarm |
| `CLEAR_THRESH` | `0.1` | Fraction of the window below which the alarm clears |
| `STALE_AFTER` | `60` | Seconds without a new frame before the camera is unavailable |
| `MQTT_HOST` / `MQTT_PORT` / `MQTT_USER` / `MQTT_PASSWORD` | `127.0.0.1` / `1883` | Broker |
| `TOPIC_PREFIX` / `DISCOVERY_PREFIX` | `firewatch` / `homeassistant` | MQTT topics |
| `FRIGATE_EVENTS` | `true` | Log detection episodes as Frigate events |
| `EVENT_GAP` | `10` | Seconds without a detection before a Frigate event is ended |

## Limitations

- This is a supplementary warning, **not a replacement for certified smoke/heat detectors**.
- The model was trained mostly on daylight colour imagery. On IR night-mode (grayscale) frames, flames lose their
  colour cue, so expect lower fire recall at night. Smoke is still detectable.
- Review the logged events regularly at first. If static objects cause repeated false positives, raise the class
  threshold or add a camera-specific mask. Detections dropped by the flame check are logged as `ignored static`.
