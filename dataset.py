import os
import cv2
import torch
from torch.utils.data import Dataset
from image_processing import compute_depth_map_advanced, enhanced_retinex

class StereoDataset(Dataset):
    """
    Dataset per immagini stereo con pre-elaborazione.
    Ottimizzato per immagini 1600x1200.
    """
    def __init__(self, left_dir, right_dir, label_file=None, augment=False, target_size=(1600, 1200)):
        """
        Inizializza il dataset.
        - left_dir: cartella con immagini della telecamera sinistra
        - right_dir: cartella con immagini della telecamera destra
        - label_file: file CSV con le etichette
        - augment: se True, applica data augmentation
        - target_size: dimensione target delle immagini (default: 1600x1200)
        """
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.augment = augment
        self.target_size = target_size
        self.left_images = sorted(os.listdir(left_dir))
        self.right_images = sorted(os.listdir(right_dir))
        self.labels = {}

        # Carica le etichette dal file CSV
        if label_file and os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    name, label = line.strip().split(',')
                    self.labels[name] = int(label)

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        """
        Restituisce un campione del dataset.
        """
        left_path = os.path.join(self.left_dir, self.left_images[idx])
        right_path = os.path.join(self.right_dir, self.right_images[idx])

        # Leggi le immagini
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        if img_left is None or img_right is None:
            raise ValueError(f"Errore nel caricamento delle immagini: {left_path} o {right_path}")

        # Ridimensiona solo se necessario
        if self.target_size is not None:
            if img_left.shape[:2][::-1] != self.target_size:
                img_left = cv2.resize(img_left, self.target_size)
            if img_right.shape[:2][::-1] != self.target_size:
                img_right = cv2.resize(img_right, self.target_size)

        # Calcola la mappa di profondità e applica il Retinex
        depth = compute_depth_map_advanced(img_left, img_right)
        enhanced_left = enhanced_retinex(img_left, depth)

        # Converte l'immagine in tensore
        enhanced_left = torch.from_numpy(enhanced_left).float().permute(2, 0, 1) / 255.0
        label = self.labels.get(self.left_images[idx], -1)

        return enhanced_left, label