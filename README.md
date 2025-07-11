#italiano
Questo codice è stato sviluppato come parte del progetto di tesi di laurea triennale in Ingegneria Informatica di Jacopo Falasco, presso l'Università degli Studi di Padova. Il lavoro si inserisce nel contesto del progetto di ricerca NAUTILUS, guidato dal professor Damiano Varagnolo.

L'obiettivo principale è il rilevamento automatico di capesante (Pecten jacobaeus) in ambiente marino, sfruttando un sistema di visione stereoscopica a basso costo. La pipeline software implementa una catena di elaborazione che include il rilevamento di oggetti (tramite l'algoritmo CenterNet), la classificazione delle immagini (con una rete ConvNeXt) e il miglioramento della qualità visiva per le immagini subacquee (attraverso un algoritmo Depth-aware Retinex).

Il sistema è stato progettato e testato per operare su piattaforme embedded a risorse limitate, come Raspberry Pi 5 e NVIDIA Jetson Nano, con lo scopo di creare uno strumento accessibile ed efficace per il monitoraggio della biodiversità marina e per supportare studi di biologia ed ecologia.

-DIRECTORY

L'organizzazione delle directory e sottodirectory è così implemenatata:

	
 stereo_vision/: Cartella principale del progetto.
  
  data/: Contiene i dati di training e validazione.

  left/: Immagini acquisite dalla telecamera sinistra.
  
  right/: Immagini acquisite dalla telecamera destra.
  labels.csv: File CSV con le etichette per ogni immagine.



models/: Cartella dove vengono salvati i modelli allenati.
  classifier.pth: Modello di classificazione allenato.



src/: Contiene il codice sorgente del progetto.
 
  stereo_ccapture.py: Gestisce l'acquisizione video stereo.
  
  image_processing.py: Contiene le funzioni per la pre-elaborazione delle immagini.
  
  dataset.py: Definisce il dataset per il training.
  
  models.py: Definisce l'architettura dei modelli di deep learning.
  
  main_train.py: Script per allenare il modello.
  
  main_inference.py: Script per eseguire l'inferenza in real-time.
  
  README.md: File che contiene istruzione e spiegazioni(il file da cui stai leggendo).

#english
This code was developed as part of the Bachelor's thesis project in Computer Engineering by Jacopo Falasco at the University of Padua, Italy. The work is a component of the NAUTILUS research project, led by Professor Damiano Varagnolo.

The primary objective is the automated detection of scallops (Pecten jacobaeus) in marine environments by leveraging a low-cost stereo vision system. The software pipeline implements a complete processing chain that includes object detection (using the CenterNet algorithm), image classification (with a ConvNeXt network), and visual quality enhancement for underwater images (through a Depth-aware Retinex algorithm).

The system is designed and tested to run on resource-constrained embedded platforms, such as the Raspberry Pi 5 and NVIDIA Jetson Nano. The ultimate goal is to provide an accessible and effective tool for monitoring marine biodiversity, supporting research in biology and ecology.

stereo_vision/: Main project folder.

  data/: Contains the training and validation data.

  left/: Images captured by the left camera.

  right/: Images captured by the right camera.

  labels.csv: CSV file with the labels for each image.


models/: Folder where the trained models are saved.

  classifier.pth: Trained classification model.


src/: Contains the project's source code.

  stereo_capture.py: Manages stereo video acquisition.

  image_processing.py: Contains functions for image preprocessing.

  dataset.py: Defines the dataset for training.

  models.py: Defines the architecture of the deep learning models.

  main_train.py: Script to train the model.

  main_inference.py: Script to run real-time inference.

README.md: File containing instructions and explanations (this file).

