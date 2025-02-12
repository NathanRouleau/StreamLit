# StreamLit - Traitement de Médias avec AWS

## 📸 Description

Ce projet utilise des services AWS pour traiter des fichiers multimédias (images et vidéos). Il permet de modérer le contenu, détecter des objets, identifier des émotions sur les visages et extraire des sous-titres et des hashtags pertinents à partir des médias. L'objectif est de proposer une solution automatisée pour l'analyse d'images et vidéos, en tirant parti des puissants outils d'Amazon Web Services (AWS) comme Rekognition, Transcribe, et Comprehend.

## Fonctionnalités

- **🚫 Modération du contenu** : Détection de contenus choquants ou inappropriés dans les images et vidéos.
- **😊 Analyse des émotions** : Détection des émotions dominantes des visages dans les images.
- **🔍 Détection des objets et des célébrités** : Identification des objets et célébrités présents sur les images.
- **📝 Extraction de texte dans les vidéos** : Utilisation d'AWS Transcribe pour convertir la voix en texte, puis extraction des mots-clés avec AWS Comprehend.
- **🔖 Génération automatique de hashtags** : Création de hashtags en fonction des objets, émotions et célébrités détectés.

## 🛠️ Prérequis

- Python 3.7+ installé
- AWS CLI configuré avec vos clés d'accès
- Installation des packages nécessaires :
  - `boto3`
  - `dotenv`
  - `opencv-python`

## 💻 Installation

1. Clonez le repository sur votre machine locale :
   ```bash
   git clone https://github.com/votre-utilisateur/project-name.git

Créez un environnement virtuel et activez-le :
  ```bash
  python3 -m venv venv
  source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
  ```

Installez les dépendances :
  ```bash
  pip install -r requirements.txt
  ```

Créez un fichier .env à la racine du projet et ajoutez vos clés d'accès AWS :
  ```bash
  ACCESS_KEY=votre_access_key_id
  SECRET_KEY=votre_secret_access_key
  ```

Assurez-vous que vos vidéos et images de test sont dans le dossier ./assets/.

# 📈 Usage
**Tester le traitement sur une image :**
- Modifiez la variable TEST_IMAGE_FILE avec le chemin de votre image de test.
- Appelez la fonction process_media en fournissant l'image et les clients AWS appropriés.

**Tester le traitement sur une vidéo :**
- Modifiez la variable TEST_VIDEO_FILE avec le chemin de votre vidéo de test.
- Appelez la fonction process_media pour traiter la vidéo et obtenir les sous-titres et hashtags.

  ```bash
  image_result = process_media(TEST_IMAGE_FILE, rekognition, transcribe, comprehend, BUCKET_NAME)
  video_result = process_media(TEST_VIDEO_FILE, rekognition, transcribe, comprehend, BUCKET_NAME)
  ```

# 📂 Structure du Dossier
  ```bash
  ├── assets/
  │   ├── tuto_jeux-video.mp4
  │   └── selfie_with_johnny-depp.png
  ├── .env
  ├── README.md
  └── process_media.py
  ```

# 📊 Exemple de Résultats
**Pour une image :**
La fonction retourne un dictionnaire de hashtags :

  ```bash
  {
    "hashtags": ["#JohnnyDepp", "#Selfie", "#Fan"]
  }
  ```

**Pour une vidéo :**
La fonction retourne des sous-titres et des hashtags extraits :

  ```bash
  {
    "subtitles": "Bienvenue dans ce tutoriel de jeu vidéo, aujourd'hui nous allons...",
    "hashtags": ["#JeuxVidéo", "#Tutoriel", "#Gaming"]
  }
  ```

# 🔍 Aide
Si vous avez besoin de plus d'informations, consultez la documentation AWS Rekognition, AWS Transcribe et AWS Comprehend.