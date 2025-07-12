import os
import torch
from torch.utils.data import DataLoader
from dataset import StereoDataset
from models import EnhancedClassifier
import time

def check_dataset_structure(left_dir, right_dir, label_file):
    """
    Verifica la struttura del dataset e la corrispondenza delle immagini.
    Returns: (bool, str) - (validità, messaggio)
    """
    if not os.path.exists(left_dir) or not os.path.exists(right_dir):
        return False, "Directory delle immagini mancanti"

    left_images = set(f for f in os.listdir(left_dir) if f.endswith(('.jpg', '.png')))
    right_images = set(f for f in os.listdir(right_dir) if f.endswith(('.jpg', '.png')))

    if len(left_images) == 0:
        return False, f"Directory {left_dir} vuota"
    if len(right_images) == 0:
        return False, f"Directory {right_dir} vuota"

    if left_images != right_images:
        missing_left = right_images - left_images
        missing_right = left_images - right_images
        msg = "Mancata corrispondenza tra immagini:\n"
        if missing_left:
            msg += f"Mancanti in left: {missing_left}\n"
        if missing_right:
            msg += f"Mancanti in right: {missing_right}"
        return False, msg

    if not os.path.exists(label_file):
        return False, f"File labels.csv non trovato in: {label_file}"

    try:
        with open(label_file, 'r') as f:
            labels = [line.strip().split(',')[0] for line in f if line.strip()]
        unlabeled_images = left_images - set(labels)
        if unlabeled_images:
            return False, f"Immagini senza etichette: {unlabeled_images}"
    except Exception as e:
        return False, f"Errore nella lettura di labels.csv: {str(e)}"

    return True, f"Dataset OK: {len(left_images)} coppie di immagini trovate"

def main():
    print("\n=== Controlli Iniziali Training ===")

    # 1. Controllo disponibilità GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilizzato: {device}")
    if device.type == 'cpu':
        print("⚠️ ATTENZIONE: Training su CPU, i tempi saranno molto più lunghi")
        proceed = input("Vuoi continuare comunque? (y/n): ").lower()
        if proceed != 'y':
            print("Training annullato")
            return

    # 2. Setup percorsi
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)

        left_dir = os.path.join(project_dir, 'data', 'left')
        right_dir = os.path.join(project_dir, 'data', 'right')
        label_file = os.path.join(project_dir, 'data', 'labels.csv')

        print(f"✓ Directory progetto: {project_dir}")
    except Exception as e:
        print(f"❌ Errore nel setup dei percorsi: {str(e)}")
        return

    # 3. Controllo dataset
    dataset_ok, dataset_msg = check_dataset_structure(left_dir, right_dir, label_file)
    if not dataset_ok:
        print(f"❌ {dataset_msg}")
        return
    print(f"✓ {dataset_msg}")

    print("\n=== Inizializzazione Training ===")

    # 4. Setup dataset e dataloader
    try:
        target_size = (1600, 1200)
        batch_size = 2

        dataset = StereoDataset(left_dir, right_dir, label_file, target_size=target_size)
        num_images = len(dataset)
        remainder = num_images % batch_size

        if remainder != 0:
            print(f"\n⚠️ ATTENZIONE: Il numero totale di immagini ({num_images}) "
                  f"non è divisibile per il batch size ({batch_size}).")
            print(f"  → {remainder} immagine/i non verranno considerate durante il training "
                  f"perché verrà usato drop_last=True.")
            print("  → Suggerimento: puoi cambiare il batch size oppure lasciare così per evitare errori di normalizzazione.\n")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        print(f"✓ Dataset caricato: {num_images} immagini")
        print(f"✓ Dimensione immagini: {target_size[0]}x{target_size[1]}")
        print(f"✓ Batch size: {batch_size} (drop_last=True)")
    except Exception as e:
        print(f"❌ Errore nel caricamento del dataset: {str(e)}")
        return

    # 5. Setup modello e ottimizzatore
    try:
        model = EnhancedClassifier(num_classes=2).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        print("✓ Modello inizializzato")
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione del modello: {str(e)}")
        return

    print("\n=== Inizio Training ===")
    print("Premi Ctrl+C per interrompere\n")

    start_training = time.time()

    try:
        for epoch in range(10):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(dataloader):
                try:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    if (batch_idx + 1) % 5 == 0:
                        print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, "
                              f"Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%")

                except Exception as e:
                    print(f"⚠️ Errore nel batch {batch_idx}: {str(e)}")
                    continue

            avg_loss = total_loss / len(dataloader)
            accuracy = 100. * correct / total
            epoch_time = time.time() - epoch_start
            print(f"\n✓ Epoch {epoch + 1} completata:")
            print(f"  - Loss media: {avg_loss:.4f}")
            print(f"  - Accuracy: {accuracy:.2f}%")
            print(f"  - Tempo impiegato: {epoch_time:.2f} secondi\n")

        total_time = time.time() - start_training
        print(f"=== Training Completato ===")
        print(f"Tempo totale di training: {total_time/60:.2f} minuti")

        # Salvataggio modello
        try:
            models_dir = os.path.join(project_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'classifier.pth')
            torch.save(model.state_dict(), model_path)
            print(f"✓ Modello salvato in: {model_path}")
        except Exception as e:
            print(f"❌ Errore nel salvataggio del modello: {str(e)}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrotto dall'utente")
        save = input("Vuoi salvare il modello attuale? (y/n): ").lower()
        if save == 'y':
            try:
                models_dir = os.path.join(project_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, 'classifier_interrupted.pth')
                torch.save(model.state_dict(), model_path)
                print(f"✓ Modello salvato in: {model_path}")
            except Exception as e:
                print(f"❌ Errore nel salvataggio del modello: {str(e)}")
    except Exception as e:
        print(f"\n❌ Errore inaspettato: {str(e)}")

    print("\n=== Sessione Terminata ===")

if __name__ == '__main__':
    main()