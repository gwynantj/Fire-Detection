# -*- coding: utf-8 -*-
"""
Fire and smoke warning service for Frigate cameras.

Polls the latest frame of each camera from the Frigate HTTP API, runs a YOLO
fire/smoke detector (ONNX), confirms detections with the temporal persistence
technique (TPT) and publishes alarms to MQTT using Home Assistant discovery.

Topics (prefix defaults to "firewatch"):
    firewatch/availability                 online/offline (LWT)
    firewatch/<camera>/availability        online/offline (frames stale/unreachable)
    firewatch/<camera>/<label>/state       ON/OFF confirmed alarm (retained)
    firewatch/<camera>/<label>/attributes  JSON: persistence ratio, best score
    firewatch/<camera>/snapshot            annotated JPEG of the latest alarm frame (retained)

Every detection episode is also logged as a Frigate event (label fire/smoke, sub label "firewatch", or
"firewatch alarm" once confirmed) with a snapshot and clip, so past detections can be reviewed in Frigate.
"""

import ast
import hashlib
import json
import logging
import os
import signal
import sys
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import requests

LOGGER = logging.getLogger('firewatch')

LABELS = ('smoke', 'fire')
COLORS = {'smoke': (200, 200, 200), 'fire': (0, 0, 255)}


def env(name, default, cast=str):
    value = os.environ.get(name)
    return default if value in (None, '') else cast(value)


class Config():

    def __init__(self):

        self.frigate_url = env('FRIGATE_URL', 'http://127.0.0.1:5000').rstrip('/')
        self.cameras = [c.strip() for c in env('CAMERAS', 'chp').split(',') if c.strip()]
        self.model_path = env('MODEL_PATH', '/models/best.onnx')
        self.imgsz = env('IMGSZ', 640, int)
        # Seconds between frames per camera
        self.interval = env('INTERVAL', 1.0, float)
        # Per-class confidence thresholds
        self.conf = {'smoke': env('SMOKE_CONF', 0.45, float), 'fire': env('FIRE_CONF', 0.40, float)}
        self.iou_thres = env('IOU_THRES', 0.45, float)
        # Ignore boxes smaller than this fraction of the frame (sensor noise, tiny lamps)
        self.min_area = env('MIN_AREA', 0.0005, float)
        # Temporal persistence technique: alarm when the fraction of frames with a detection
        # in the last WINDOW_SIZE frames exceeds PERSISTENCE_THRESH, clear when it drops below CLEAR_THRESH
        self.window_size = env('WINDOW_SIZE', 20, int)
        self.persistence_thresh = env('PERSISTENCE_THRESH', 0.5, float)
        self.clear_thresh = env('CLEAR_THRESH', 0.1, float)
        # Mark a camera unavailable if no new frame arrives for this many seconds
        self.stale_after = env('STALE_AFTER', 60, float)
        # Re-publish the snapshot this often while an alarm is active
        self.snapshot_interval = env('SNAPSHOT_INTERVAL', 10, float)
        self.mqtt_host = env('MQTT_HOST', '127.0.0.1')
        self.mqtt_port = env('MQTT_PORT', 1883, int)
        self.mqtt_user = env('MQTT_USER', None)
        self.mqtt_password = env('MQTT_PASSWORD', None)
        self.topic_prefix = env('TOPIC_PREFIX', 'firewatch')
        self.discovery_prefix = env('DISCOVERY_PREFIX', 'homeassistant')
        # Log detection episodes as Frigate events, ended after EVENT_GAP seconds without a detection
        self.frigate_events = env('FRIGATE_EVENTS', 'true').lower() in ('1', 'true', 'yes')
        self.event_gap = env('EVENT_GAP', 10, float)


class Detector():

    def __init__(self, model_path, imgsz, iou_thres):

        options = ort.SessionOptions()
        # One frame per second needs few threads, and spin-waiting between frames wastes a core
        options.intra_op_num_threads = 2
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.iou_thres = iou_thres
        names = self.session.get_modelmeta().custom_metadata_map.get('names')
        # Ultralytics stores names as a python dict literal, e.g. "{0: 'smoke', 1: 'fire'}"
        self.names = ast.literal_eval(names) if names else dict(enumerate(LABELS))

    def letterbox(self, im):

        h, w = im.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, left, top

    def __call__(self, im0, conf):

        im, r, left, top = self.letterbox(im0)
        blob = cv2.dnn.blobFromImage(im, 1 / 255.0, swapRB=True)
        # Output (1, 4 + nc, anchors): cx, cy, w, h followed by class scores
        pred = self.session.run(None, {self.input_name: blob})[0][0].T

        scores = pred[:, 4:]
        class_ids = scores.argmax(axis=1)
        confs = scores[np.arange(len(scores)), class_ids]
        min_conf = min(conf.values())
        keep = confs >= min_conf
        pred, class_ids, confs = pred[keep], class_ids[keep], confs[keep]

        h0, w0 = im0.shape[:2]
        detections = list()
        for c in np.unique(class_ids):
            label = self.names[int(c)]
            idx = np.where((class_ids == c) & (confs >= conf.get(label, min_conf)))[0]
            if not len(idx):
                continue
            # Undo letterbox and convert cx, cy, w, h to x, y, w, h in original pixels
            boxes = pred[idx, :4].copy()
            boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - left) / r
            boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - top) / r
            boxes[:, 2] /= r
            boxes[:, 3] /= r
            nms = cv2.dnn.NMSBoxes(boxes.tolist(), confs[idx].tolist(), conf.get(label, min_conf), self.iou_thres)
            for i in np.array(nms).flatten():
                x, y, w, h = boxes[i]
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(w0, int(x + w)), min(h0, int(y + h))
                detections.append({'label': label, 'conf': float(confs[idx][i]), 'box': (x1, y1, x2, y2)})

        return detections


class Persistence():
    """Temporal persistence technique (TPT) with hysteresis, one sliding window per label."""

    def __init__(self, window_size, thresh, clear_thresh):

        self.buffer = deque(maxlen=window_size)
        self.thresh = thresh
        self.clear_thresh = clear_thresh
        self.active = False

    def update(self, detected):

        self.buffer.append(bool(detected))
        ratio = sum(self.buffer) / self.buffer.maxlen
        if not self.active and ratio > self.thresh:
            self.active = True
        elif self.active and ratio < self.clear_thresh:
            self.active = False
        return self.active, ratio


class Camera():

    def __init__(self, name, cfg):

        self.name = name
        self.cfg = cfg
        self.url = f'{cfg.frigate_url}/api/{name}/latest.jpg'
        self.windows = {label: Persistence(cfg.window_size, cfg.persistence_thresh, cfg.clear_thresh) for label in LABELS}
        self.best = {label: 0.0 for label in LABELS}
        self.frame_hash = None
        self.last_new_frame = time.monotonic()
        self.available = None
        self.last_snapshot = 0.0
        # Open Frigate event id and time of the last detection, per label
        self.events = {label: None for label in LABELS}
        self.last_seen = {label: 0.0 for label in LABELS}

    def fetch(self, session):

        response = session.get(self.url, timeout=10)
        response.raise_for_status()
        digest = hashlib.md5(response.content).digest()
        if digest != self.frame_hash:
            self.frame_hash = digest
            self.last_new_frame = time.monotonic()
        im = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError('could not decode frame')
        return im

    def stale(self):

        return time.monotonic() - self.last_new_frame > self.cfg.stale_after


class FireWatch():

    def __init__(self, cfg):

        self.cfg = cfg
        self.detector = Detector(cfg.model_path, cfg.imgsz, cfg.iou_thres)
        self.cameras = [Camera(name, cfg) for name in cfg.cameras]
        self.http = requests.Session()
        self.running = True

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f'{cfg.topic_prefix}-{os.getpid()}')
        if cfg.mqtt_user:
            self.client.username_pw_set(cfg.mqtt_user, cfg.mqtt_password)
        self.client.will_set(self.topic('availability'), 'offline', retain=True)
        self.client.on_connect = self.on_connect

    def topic(self, *parts):

        return '/'.join((self.cfg.topic_prefix,) + parts)

    def on_connect(self, client, userdata, flags, reason_code, properties):

        LOGGER.info('Connected to MQTT broker (%s)', reason_code)
        self.publish_discovery()
        client.publish(self.topic('availability'), 'online', retain=True)
        # Force camera availability and alarm states to be re-published after a reconnect
        for cam in self.cameras:
            cam.available = None
            for label in LABELS:
                self.publish_state(cam, label, cam.windows[label].active, 0.0)

    def publish_discovery(self):

        device = {
            'identifiers': [self.cfg.topic_prefix],
            'name': 'Fire Watch',
            'manufacturer': 'Fire-Detection',
            'model': os.path.basename(self.cfg.model_path),
        }
        for cam in self.cameras:
            availability = [{'topic': self.topic('availability')}, {'topic': self.topic(cam.name, 'availability')}]
            for label, device_class in (('fire', 'safety'), ('smoke', 'smoke')):
                uid = f'{self.cfg.topic_prefix}_{cam.name}_{label}'
                payload = {
                    'name': f'{cam.name} {label}',
                    'unique_id': uid,
                    'default_entity_id': f'binary_sensor.{uid}',
                    'device_class': device_class,
                    'state_topic': self.topic(cam.name, label, 'state'),
                    'json_attributes_topic': self.topic(cam.name, label, 'attributes'),
                    'availability': availability,
                    'availability_mode': 'all',
                    'device': device,
                }
                self.client.publish(f'{self.cfg.discovery_prefix}/binary_sensor/{uid}/config', json.dumps(payload), retain=True)

            uid = f'{self.cfg.topic_prefix}_{cam.name}_snapshot'
            payload = {
                'name': f'{cam.name} fire snapshot',
                'unique_id': uid,
                'default_entity_id': f'camera.{uid}',
                'topic': self.topic(cam.name, 'snapshot'),
                'availability': availability[:1],
                'device': device,
            }
            self.client.publish(f'{self.cfg.discovery_prefix}/camera/{uid}/config', json.dumps(payload), retain=True)

    def publish_state(self, cam, label, active, ratio):

        self.client.publish(self.topic(cam.name, label, 'state'), 'ON' if active else 'OFF', retain=True)
        attributes = {'persistence': round(ratio, 2), 'best_score': round(cam.best[label], 2), 'camera': cam.name}
        self.client.publish(self.topic(cam.name, label, 'attributes'), json.dumps(attributes), retain=True)

    def set_available(self, cam, available):

        if cam.available != available:
            cam.available = available
            self.client.publish(self.topic(cam.name, 'availability'), 'online' if available else 'offline', retain=True)
            LOGGER.log(logging.INFO if available else logging.WARNING, '%s: camera %s', cam.name, 'online' if available else 'offline')

    def event_start(self, cam, label, found, w, h):

        draw = [{'box': [d['box'][0] / w, d['box'][1] / h, (d['box'][2] - d['box'][0]) / w, (d['box'][3] - d['box'][1]) / h],
                 'color': list(COLORS[label]), 'score': round(d['conf'], 2)} for d in found]
        body = {'sub_label': 'firewatch', 'score': max(d['conf'] for d in found), 'duration': None,
                'include_recording': True, 'draw': {'boxes': draw}}
        try:
            response = self.http.post(f'{self.cfg.frigate_url}/api/events/{cam.name}/{label}/create', json=body, timeout=5)
            response.raise_for_status()
            cam.events[label] = response.json()['event_id']
            LOGGER.info('%s: %s event %s started', cam.name, label, cam.events[label])
        except Exception as e:
            LOGGER.warning('%s: could not create Frigate %s event: %s', cam.name, label, e)

    def event_sub_label(self, cam, label, sub_label):

        try:
            self.http.post(f'{self.cfg.frigate_url}/api/events/{cam.events[label]}/sub_label', json={'subLabel': sub_label}, timeout=5).raise_for_status()
        except Exception as e:
            LOGGER.warning('%s: could not label Frigate event %s: %s', cam.name, cam.events[label], e)

    def event_end(self, cam, label):

        try:
            self.http.put(f'{self.cfg.frigate_url}/api/events/{cam.events[label]}/end', json={}, timeout=5).raise_for_status()
            LOGGER.info('%s: %s event %s ended', cam.name, label, cam.events[label])
        except Exception as e:
            LOGGER.warning('%s: could not end Frigate event %s: %s', cam.name, cam.events[label], e)
        cam.events[label] = None

    def annotate(self, im, detections):

        im = im.copy()
        for d in detections:
            x1, y1, x2, y2 = d['box']
            color = COLORS.get(d['label'], (0, 255, 255))
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 3)
            cv2.putText(im, f"{d['label']} {d['conf']:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(im, time.strftime('%Y-%m-%d %H:%M:%S'), (10, im.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return im

    def process(self, cam):

        try:
            im = cam.fetch(self.http)
        except Exception as e:
            LOGGER.warning('%s: frame fetch failed: %s', cam.name, e)
            if cam.stale():
                self.set_available(cam, False)
            return
        # Frigate keeps serving the last frame when a camera drops, so an unchanged image means a dead feed
        if cam.stale():
            self.set_available(cam, False)
            return
        self.set_available(cam, True)

        h, w = im.shape[:2]
        detections = [d for d in self.detector(im, self.cfg.conf)
                      if (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]) >= self.cfg.min_area * w * h]

        now = time.monotonic()
        alarm = False
        for label in LABELS:
            found = [d for d in detections if d['label'] == label]
            window = cam.windows[label]
            was_active = window.active
            if found:
                cam.best[label] = max(cam.best[label], max(d['conf'] for d in found))
                cam.last_seen[label] = now
                if self.cfg.frigate_events and cam.events[label] is None:
                    self.event_start(cam, label, found, w, h)
            active, ratio = window.update(found)
            alarm |= active
            if cam.events[label] is not None:
                if active and not was_active:
                    self.event_sub_label(cam, label, 'firewatch alarm')
                elif not active and now - cam.last_seen[label] > self.cfg.event_gap:
                    self.event_end(cam, label)
            if active != was_active:
                LOGGER.warning('%s: %s alarm %s (persistence %.2f, best %.2f)', cam.name, label, 'RAISED' if active else 'cleared', ratio, cam.best[label])
                if active:
                    # Snapshot goes out before the state so automations triggered by ON can attach it
                    self.client.publish(self.topic(cam.name, 'snapshot'), cv2.imencode('.jpg', self.annotate(im, detections))[1].tobytes(), retain=True)
                    cam.last_snapshot = now
                self.publish_state(cam, label, active, ratio)
                if not active:
                    cam.best[label] = 0.0
            elif active:
                self.publish_state(cam, label, active, ratio)

        if detections:
            LOGGER.info('%s: %s', cam.name, ', '.join(f"{d['label']} {d['conf']:.2f}" for d in detections))

        if alarm and detections and now - cam.last_snapshot >= self.cfg.snapshot_interval:
            self.client.publish(self.topic(cam.name, 'snapshot'), cv2.imencode('.jpg', self.annotate(im, detections))[1].tobytes(), retain=True)
            cam.last_snapshot = now

    def stop(self, *args):

        self.running = False

    def run(self):

        self.client.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=30)
        self.client.loop_start()
        LOGGER.info('Watching %s every %.1fs with %s', ', '.join(self.cfg.cameras), self.cfg.interval, self.cfg.model_path)

        while self.running:
            started = time.monotonic()
            for cam in self.cameras:
                self.process(cam)
            time.sleep(max(0.0, self.cfg.interval - (time.monotonic() - started)))

        for cam in self.cameras:
            for label in LABELS:
                if cam.events[label] is not None:
                    self.event_end(cam, label)
        self.client.publish(self.topic('availability'), 'offline', retain=True).wait_for_publish(5)
        self.client.loop_stop()
        self.client.disconnect()


def main():

    logging.basicConfig(level=env('LOG_LEVEL', 'INFO'), format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    watch = FireWatch(Config())
    signal.signal(signal.SIGTERM, watch.stop)
    signal.signal(signal.SIGINT, watch.stop)
    watch.run()


if __name__ == '__main__':
    main()
