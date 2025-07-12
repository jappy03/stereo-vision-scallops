import cv2
import numpy as np

def compute_depth_map_advanced(img_left, img_right):
    """
    Calcola una mappa di profondità avanzata dalle immagini stereo.
    - img_left: immagine della telecamera sinistra.
    - img_right: immagine della telecamera destra.
    Returns: mappa di profondità normalizzata.
    """
    # Converti le immagini in scala di grigi
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Equalizzazione dell'istogramma per migliorare il contrasto
    gray_left = cv2.equalizeHist(gray_left)
    gray_right = cv2.equalizeHist(gray_right)

    # Configura il calcolo della disparità con StereoBM
    stereo = cv2.StereoBM_create(numDisparities=16*10, blockSize=21)
    stereo.setMinDisparity(0)
    stereo.setUniquenessRatio(10)
    stereo.setSpeckleWindowSize(100)
    stereo.setSpeckleRange(32)
    stereo.setDisp12MaxDiff(1)

    # Calcola la mappa di disparità
    disparity = stereo.compute(gray_left, gray_right)

    # Normalizza la mappa di disparità
    disparity = cv2.medianBlur(disparity, 5)
    disparity = cv2.normalize(disparity, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return disparity

def enhanced_retinex(image, depth, sigma_list=[15, 80, 250]):
    """
    Applica il Retinex migliorato con depth-awareness.
    - image: immagine originale.
    - depth: mappa di profondità.
    - sigma_list: lista di valori sigma per il Retinex multi-scala.
    Returns: immagine migliorata.
    """
    image = image.astype(np.float32) + 1.0
    retinex = np.zeros_like(image)
    weights = [1/len(sigma_list)] * len(sigma_list)

    # Applica il Retinex multi-scala
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(image, (0, 0), sigma)
        retinex += weights[sigma_list.index(sigma)] * (np.log10(image) - np.log10(blur))

    # Normalizza la mappa di profondità
    depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

    # Combina Retinex e informazione di profondità
    retinex = retinex * (1 + depth_norm[:, :, np.newaxis])
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255
    retinex = np.clip(retinex, 0, 255).astype(np.uint8)

    return retinex