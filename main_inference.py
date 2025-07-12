import cv2
import torch
import os
from datetime import datetime, timedelta
import numpy as np
from stereo_capture import StereoVideoProcessor
from models import EnhancedClassifier
from image_processing import compute_depth_map_advanced, enhanced_retinex

def compute_frame_similarity(frame1, frame2):
    """
    Calcola la similarità tra due frame usando MSE (Mean Squared Error).
    Returns: True se i frame sono simili, False altrimenti
    """
    if frame1 is None or frame2 is None:
        return False

    # Ridimensiona i frame per velocizzare il confronto
    size = (160, 120)  # Risoluzione ridotta per il confronto
    f1 = cv2.resize(frame1, size)
    f2 = cv2.resize(frame2, size)

    # Calcola MSE
    mse = np.mean((f1 - f2) ** 2)
    # Soglia di similarità (da regolare in base alle tue necessità)
    return mse < 1000  # Soglia da calibrare

def check_cameras():
    """
    Verifica la disponibilità delle telecamere.
    Returns: (bool, str) - (disponibilità, messaggio)
    """
    try:
        cap1 = cv2.VideoCapture(0)
        cap2 = cv2.VideoCapture(1)

        if not cap1.isOpened() and not cap2.isOpened():
            return False, "Nessuna telecamera trovata"
        elif not cap1.isOpened():
            return False, "Telecamera sinistra non trovata"
        elif not cap2.isOpened():
            return False, "Telecamera destra non trovata"

        cap1.release()
        cap2.release()
        return True, "Telecamere OK"
    except Exception as e:
        return False, f"Errore nell'accesso alle telecamere: {str(e)}"

def check_model(model_path, device):
    """
    Verifica l'esistenza e la validità del modello.
    Returns: (bool, str) - (disponibilità, messaggio)
    """
    if not os.path.exists(model_path):
        return False, f"Modello non trovato in: {model_path}"

    try:
        # Prova a caricare il modello
        torch.load(model_path, map_location=device)
        return True, "Modello OK"
    except Exception as e:
        return False, f"Errore nel caricamento del modello: {str(e)}"

def main():
    print("\n=== Controlli Iniziali ===")

    # 1. Controllo disponibilità GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilizzato: {device}")
    if device.type == 'cpu':
        print("⚠️ ATTENZIONE: GPU non disponibile, le performance potrebbero essere ridotte")

    # 2. Controllo presenza modello
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'classifier.pth')
    model_ok, model_msg = check_model(model_path, device)
    if not model_ok:
        print(f"❌ {model_msg}")
        return
    print(f"✓ {model_msg}")

    # 3. Controllo telecamere
    cameras_ok, cameras_msg = check_cameras()
    if not cameras_ok:
        print(f"❌ {cameras_msg}")
        return
    print(f"✓ {cameras_msg}")

    print("\n=== Inizializzazione Sistema ===")

    # Carica il modello
    try:
        model = EnhancedClassifier(num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✓ Modello caricato correttamente")
    except Exception as e:
        print(f"❌ Errore nel caricamento del modello: {str(e)}")
        return

    # Crea cartella per i frame delle capesante
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'detected_scallops', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"✓ I frame verranno salvati in: {save_dir}")

    # Inizializza il processore video
    try:
        video_processor = StereoVideoProcessor()
        video_processor.start_capture()
        print("✓ Processore video inizializzato")
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione del video processor: {str(e)}")
        return

    # Variabili per gestire il salvataggio dei frame
    last_saved_frame = None
    last_detection_time = None
    detection_counter = 0
    min_time_between_saves = timedelta(seconds=2)  # Tempo minimo tra due salvataggi

    print("\n=== Sistema Pronto ===")
    print("Premi 'q' per uscire")

    try:
        while True:
            left_frame, right_frame = video_processor.get_stereo_frames()
            if left_frame is None or right_frame is None:
                print("⚠️ Frame non disponibile, riprovo...")
                continue

            # Pre-elaborazione
            try:
                depth = compute_depth_map_advanced(left_frame, right_frame)
                enhanced_left = enhanced_retinex(left_frame, depth)
            except Exception as e:
                print(f"⚠️ Errore nella pre-elaborazione: {str(e)}")
                continue

            # Prepara il tensore
            try:
                input_tensor = torch.from_numpy(enhanced_left).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            except Exception as e:
                print(f"⚠️ Errore nella preparazione del tensore: {str(e)}")
                continue

            # Inferenza
            try:
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    is_scallop = predicted.item() == 1
                    label = "Capesante" if is_scallop else "Altro"
            except Exception as e:
                print(f"⚠️ Errore nell'inferenza: {str(e)}")
                continue

            # Gestione salvataggio frame con capesante
            current_time = datetime.now()
            if is_scallop:
                should_save = False

                if last_detection_time is None or \
                   (current_time - last_detection_time) > min_time_between_saves:

                    if last_saved_frame is None or \
                       not compute_frame_similarity(left_frame, last_saved_frame):
                        should_save = True

                if should_save:
                    try:
                        # Salva il frame
                        detection_counter += 1
                        filename = f"scallop_{current_time.strftime('%Y%m%d_%H%M%S')}_{detection_counter}.jpg"
                        save_path = os.path.join(save_dir, filename)

                        # Salva sia il frame originale che quello elaborato
                        cv2.imwrite(save_path, left_frame)
                        enhanced_path = os.path.join(save_dir, f"enhanced_{filename}")
                        cv2.imwrite(enhanced_path, enhanced_left)

                        print(f"✓ Salvato frame con capesanta: {filename}")

                        # Aggiorna le variabili di tracking
                        last_saved_frame = left_frame.copy()
                        last_detection_time = current_time
                    except Exception as e:
                        print(f"⚠️ Errore nel salvataggio del frame: {str(e)}")

            # Visualizza il risultato
            info_text = f"{label} - Rilevamenti: {detection_counter}"
            cv2.putText(left_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mostra anche la mappa di profondità
            depth_display = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            # Mostra i frame
            cv2.imshow('Stereo Inference', left_frame)
            cv2.imshow('Depth Map', depth_colored)

            # Esci con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⚠️ Interruzione da tastiera")
    except Exception as e:
        print(f"\n❌ Errore inaspettato: {str(e)}")
    finally:
        video_processor.stop()
        cv2.destroyAllWindows()
        print(f"\n=== Sessione Terminata ===")
        print(f"✓ Totale capesante rilevate e salvate: {detection_counter}")
        print(f"✓ I frame sono stati salvati in: {save_dir}")

if __name__ == '__main__':
    main()