import argparse
import cv2
import numpy as np
import threading
import queue
import multiprocessing
import time
import logging
import os
import sys


os.makedirs('log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/task4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")

class SensorX(Sensor):
    '''Sensor X'''
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data

class SensorCam(Sensor):
    def __init__(self, cam_id: int, resolution: tuple, stop_event=None):
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.cap = cv2.VideoCapture(cam_id)
        self.stop_event = stop_event
        
        if not self.cap.isOpened():
            logger.error(f"Камера {cam_id} не найдена или недоступна.")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        logger.info(f"Camera {cam_id} started: {resolution[0]}x{resolution[1]}")

    def _loop(self):
        error_count = 0
        max_errors = 10

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                error_count += 1

                if error_count > max_errors:
                    logger.error(f"Camera disconnected or failed after {error_count} errors. Shutting down...")
                    self.running = False

                    if self.stop_event:
                        self.stop_event.set()
                    break

                if error_count % 5 == 0:
                    logging.warning(f"Camera read error ({error_count}/{max_errors}). Check connection.")
                time.sleep(0.1)
                continue
            error_count = 0

            try:
                self.q.put_nowait((time.time(), frame))
            except queue.Full:
                try:
                    self.q.get_nowait()
                    self.q.put_nowait((time.time(), frame))
                except queue.Empty:
                    pass

    def get(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None, None

    def __del__(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        logger.info("Camera released.")

class WindowImage:
    def __init__(self, fps: float):
        self.delay = int(1000 / fps)
        self.name = "Task4 Output"
        try:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
            logger.info(f"Window created. FPS: {fps}")
        except Exception as e:
            logger.error(f"Window creation failed: {e}")
            sys.exit(1)

    def show(self, img):
        if img is not None:
            cv2.imshow(self.name, img)

    def __del__(self):
        try:
            cv2.destroyWindow(self.name)
        except:
            pass
        logger.info("Window destroyed.")

def sensor_worker(delay, data_queue, stop_event):
    """Создает SensorX внутри, чтобы изолировать ресурсы в потоке/процессе"""
    sensor = SensorX(delay)
    while not stop_event.is_set():
        try:
            val = sensor.get()
            packet = {'value': val, 'time': time.time()}
            try:
                data_queue.put_nowait(packet)
            except queue.Full:
                try:
                    data_queue.get_nowait()
                    data_queue.put_nowait(packet)
                except queue.Empty:
                    pass
        except Exception as e:
            logger.error(f"Worker error: {e}")
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='ID камеры (int)')
    parser.add_argument('--resolution', type=str, default='1280x720', help='Разрешение WxH')
    parser.add_argument('--display_freq', type=float, default=30.0, help='Частота отображения')
    parser.add_argument('--mode', choices=['thread', 'process'], default='thread', help='Режим работы')
    args = parser.parse_args()

    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        logger.error("Invalid format. Use: WxH")
        sys.exit(1)

    if args.mode == 'process':
        stop_event = multiprocessing.Event()
    else:
        stop_event = threading.Event()
    cam = SensorCam(args.camera, resolution, stop_event)
    win = WindowImage(args.display_freq)

    delays = [0.01, 0.1, 1.0]
    queues = []
    workers = []

    Q = multiprocessing.Queue if args.mode == 'process' else queue.Queue
    WorkerClass = multiprocessing.Process if args.mode == 'process' else threading.Thread

    for i, d in enumerate(delays):
        q = Q(maxsize=1)
        queues.append(q)
        worker_obj = WorkerClass(target=sensor_worker, args=(d, q, stop_event), daemon=True)
        worker_obj.start()
        workers.append(worker_obj)
        logger.info(f"Started SensorX_{i+1} ({1/d}Hz) in {args.mode}")

    last_frame = None
    last_sensors = [None] * 3

    try:
        while True:
            if stop_event.is_set():
                logger.info(f"Camera {args.camera} stopped.")
                break

            _, frame = cam.get()
            if frame is not None:
                last_frame = frame.copy()

            img = last_frame.copy() if last_frame is not None else np.zeros((height, width, 3), dtype=np.uint8)
            now = time.time()

            # Датчики
            for i, q in enumerate(queues):
                try:
                    packet = q.get_nowait()
                    last_sensors[i] = packet
                except queue.Empty:
                    pass

                if last_sensors[i]:
                    delay_ms = (now - last_sensors[i]['time']) * 1000
                    txt = f"S{i+1}({int(1/delays[i])}Hz): {last_sensors[i]['value']} | Delay: {delay_ms:.1f}ms"
                    cv2.putText(img, txt, (10, 50 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(img, f"Mode: {args.mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            win.show(img)

            if cv2.waitKey(win.delay) & 0xFF == ord('q'):
                logger.info("Exit requested")
                break
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Shutting down all resources...")
        stop_event.set()
        for worker_obj in workers:
            worker_obj.join(timeout=2.0)
        del cam
        del win
        logger.info("Done.")

if __name__ == '__main__':
    main()