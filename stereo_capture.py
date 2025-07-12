import cv2
import queue
import threading
import os
from datetime import datetime

class StereoVideoProcessor:
    """
    Classe per la gestione dell'acquisizione video stereo in real-time.
    Permette di acquisire frame sincronizzati da due telecamere o video.
    """
    def __init__(self, left_source=0, right_source=1, buffer_size=10):
        """
        Inizializza il processore video stereo.
        - left_source, right_source: indici delle telecamere o percorsi video.
        - buffer_size: dimensione del buffer per i frame.
        """
        self.left_cap = cv2.VideoCapture(left_source)
        self.right_cap = cv2.VideoCapture(right_source)

        # Verifica apertura degli stream video
        if not self.left_cap.isOpened() or not self.right_cap.isOpened():
            raise ValueError("Errore nell'apertura degli stream video")

        # Imposta le stesse proprietà per entrambe le camere
        for cap in [self.left_cap, self.right_cap]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Larghezza del frame
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Altezza del frame
            cap.set(cv2.CAP_PROP_FPS, 30)  # Frame per secondo

        # Code thread-safe per i frame
        self.left_queue = queue.Queue(maxsize=buffer_size)
        self.right_queue = queue.Queue(maxsize=buffer_size)

        # Flag per il controllo dei thread
        self.is_running = False
        self.save_frames = False

        # Cartelle per il salvataggio dei frame
        self.save_dir = 'captured_frames'
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'left'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'right'), exist_ok=True)

    def start_capture(self):
        """
        Avvia i thread di cattura per entrambe le telecamere.
        """
        self.is_running = True
        self.left_thread = threading.Thread(target=self._capture_frames,args=(self.left_cap, self.left_queue, 'left'))
        self.right_thread = threading.Thread(target=self._capture_frames,args=(self.right_cap, self.right_queue, 'right'))

        self.left_thread.daemon = True
        self.right_thread.daemon = True

        self.left_thread.start()
        self.right_thread.start()

    def _capture_frames(self, cap, frame_queue, side):
        """
        Thread worker per la cattura dei frame.
        - cap: oggetto VideoCapture.
        - frame_queue: coda per i frame catturati.
        - side: 'left' o 'right' per identificare la telecamera.
        """
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Errore nella lettura del frame dalla camera {side}")
                continue

            # Se la coda è piena, rimuovi il frame più vecchio
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                frame_queue.put(frame, timeout=0.1)

                # Salva i frame se richiesto
                if self.save_frames:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    filename = f"{side}_{timestamp}.jpg"
                    save_path = os.path.join(self.save_dir, side, filename)
                    cv2.imwrite(save_path, frame)
            except queue.Full:
                continue

    def get_stereo_frames(self, timeout=1.0):
        """
        Ottiene una coppia di frame sincronizzati dalle telecamere.
        Returns: (left_frame, right_frame) o (None, None) se timeout.
        """
        try:
            left_frame = self.left_queue.get(timeout=timeout)
            right_frame = self.right_queue.get(timeout=timeout)
            return left_frame, right_frame
        except queue.Empty:
            return None, None

    def toggle_save_frames(self):
        """
        Attiva/disattiva il salvataggio dei frame.
        """
        self.save_frames = not self.save_frames
        print(f"Salvataggio frame {'attivato' if self.save_frames else 'disattivato'}")

    def stop(self):
        """
        Ferma l'acquisizione e rilascia le risorse.
        """
        self.is_running = False
        if hasattr(self, 'left_thread'):
            self.left_thread.join()
        if hasattr(self, 'right_thread'):
            self.right_thread.join()
        self.left_cap.release()
        self.right_cap.release()