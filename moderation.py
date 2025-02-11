import os
from matplotlib import pyplot as plt
import cv2

def check_filetype(filename):
    """
    Détermine le type de fichier en fonction de son extension.

    Cette fonction prend un nom de fichier en entrée, extrait son extension et détermine
    le type de fichier (par exemple, image, vidéo). Si l'extension du fichier est reconnue comme un format
    d'image courant (jpg, png, tiff, svg) ou un format de vidéo courant (mp4, avi, mkv), elle attribue
    le type correspondant. Sinon, le type de fichier est défini sur None.

    Paramètres :
    - filename (str) : Le chemin vers le fichier incluant le nom de fichier.

    Retourne :
    - str ou None : Le type de fichier déterminé ('image', 'vidéo') ou None si le type de fichier
      n'est pas reconnu.

    Exemple :
    >>> check_filetype("/chemin/vers/image.jpg")
    'image'
    >>> check_filetype("/chemin/vers/video.mp4")
    'vidéo'
    >>> check_filetype("/chemin/vers/fichierinconnu.xyz")
    None
    """

    # Extrait le nom de base du fichier à partir du chemin de fichier fourni.
    file_basename = os.path.basename(filename)

    # Sépare le nom de base sur le point et prend la dernière partie comme extension.
    extension = file_basename.split(".")[-1]

    # Détermine le type de fichier en fonction de l'extension.
    if extension in ["jpg", "png", "tiff", "svg"]:
        filetype = "image"
    elif extension in ["mp4", "avi", "mkv"]:
        filetype = "vidéo"
    else:
        filetype = None

    # Enregistre le type de fichier détecté.
    print(f"[INFO] : Le fichier {file_basename} est de type : {filetype}")
    
    return filetype


def extract_frame_video(video_path, frame_id):
    """
    Extrait une image spécifique d'une vidéo.

    Cette fonction utilise OpenCV pour ouvrir une vidéo à partir du chemin spécifié et extrait une image
    particulière en fonction de son ID. L'ID de l'image correspond à l'ordre de l'image dans la vidéo, en commençant
    par 0 pour la première image. Si l'extraction réussit, l'image est retournée sous forme d'un tableau Numpy.

    Paramètres :
    - video_path (str) : Le chemin vers le fichier vidéo d'où extraire l'image.
    - frame_id (int) : L'identifiant (ID) de l'image à extraire.

    Retourne :
    - ndarray ou None : L'image extraite (un tableau Numpy) si l'extraction est réussie,
      sinon `None`.

    Exemple :
    >>> image = extract_frame_video("/chemin/vers/video.mp4", 150)
    >>> type(image)
    <class 'numpy.ndarray'>
    """

    # Ouvre la vidéo à partir du chemin fourni.
    video = cv2.VideoCapture(video_path)

    # Positionne le lecteur vidéo sur l'image spécifiée par frame_id.
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    # Lit l'image actuelle.
    ret, image = video.read()

    # Si la lecture réussit (ret est True), retourne l'image.
    # Sinon, retourne None.
    return image if ret else None







import os, boto3
from dotenv import load_dotenv

def get_aws_session():
    """
    Crée et retourne une session AWS.

    Cette fonction charge les variables d'environnement depuis un fichier .env situé dans le répertoire
    courant ou les parents de celui-ci, récupère les clés d'accès AWS (`ACCESS_KEY` et `SECRET_KEY`),
    et initialise une session AWS avec ces identifiants ainsi qu'avec une région spécifiée (dans cet exemple,
    'us-east-1'). Elle est particulièrement utile pour configurer une session AWS de manière sécurisée sans
    hardcoder les clés d'accès dans le code.

    Retourne :
    - Session : Un objet session de boto3 configuré avec les clés d'accès et la région AWS.

    Exemple d'utilisation :
    >>> session_aws = get_aws_session()
    >>> type(session_aws)
    <class 'boto3.session.Session'>
    """

    # Charge les variables d'environnement depuis .env.
    load_dotenv()

    # Crée une session AWS avec les clés d'accès et la région définies dans les variables d'environnement.
    aws_session = boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY"),        # Récupère l'ID de clé d'accès depuis les variables d'environnement.
        aws_secret_access_key=os.getenv("SECRET_KEY"),    # Récupère la clé d'accès secrète depuis les variables d'environnement.
        region_name="us-east-1"                           # Spécifie la région AWS à utiliser.
    )
    
    # Retourne l'objet session créé.
    return aws_session




def moderate_image(image_path, aws_service):
    """
    Détecte du contenu nécessitant une modération dans une image en utilisant un service AWS spécifié.

    Cette fonction ouvre une image depuis un chemin donné, puis utilise le service AWS (comme Amazon Rekognition)
    pour détecter les contenus potentiellement inappropriés ou sensibles (comme la nudité, la violence, etc.).
    Elle collecte et retourne une liste des étiquettes de modération identifiées pour cette image.

    Paramètres :
    - image_path (str) : Le chemin vers l'image à analyser.
    - aws_service (object) : Un objet de service AWS configuré, capable de réaliser des opérations de détection
      de contenu nécessitant une modération (par exemple, un client Amazon Rekognition).

    Retourne :
    - list[str] : Une liste des noms des étiquettes de modération détectées pour l'image.

    Exemple d'utilisation :
    >>> aws_rekognition_client = boto3.client('rekognition', region_name='us-east-1')
    >>> moderate_image("/chemin/vers/image.jpg", aws_rekognition_client)
    ['Nudity', 'Explicit Violence']
    """
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    # Appeler le service AWS (ici Rekognition) pour analyser l'image
    response = aws_service.detect_moderation_labels(
        Image={'Bytes': image_bytes}
    )

    # Extraire les étiquettes de modération de la réponse
    moderation_labels = response.get('ModerationLabels', [])
    inappropriate_themes = [
        label['Name'] for label in moderation_labels if label['Confidence'] > 80
    ]

    if(inappropriate_themes):
        return inappropriate_themes
    else:
        return None






import os
import boto3
from dotenv import load_dotenv
def get_aws_session():
    """
    Crée et retourne une session AWS.
    Charge les clés d'accès depuis un fichier .env pour une configuration sécurisée.
    """
    # Charger les variables d'environnement à partir du fichier .env
    load_dotenv()

    # Créer une session AWS avec les clés d'accès et la région spécifiée
    return boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY"),
        region_name="us-east-1"  # Vous pouvez changer la région si nécessaire
    )

        


import boto3
import os
from dotenv import load_dotenv

def get_aws_session():
    # Charger les variables d'environnement depuis le fichier .env
    load_dotenv()

    # Créer une session AWS avec les clés d'accès et la région spécifiée
    return boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY"),
    ) 

def create_s3_bucket(bucket_name):
    # Créer une session AWS
    aws_session = get_aws_session()

    # Créer un client S3
    s3_client = aws_session.client("s3")

    # Créer le bucket S3
    try:
        response = s3_client.create_bucket(
            Bucket=bucket_name,
        )
        print(f"Le bucket S3 '{bucket_name}' a été créé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la création du bucket S3 : {e}")




import os
import time
import urllib.request
import json
import time

def generate_unique_job_name(base_name="transcription-job"):
    """
    Génère un nom de travail unique en utilisant l'horodatage.
    """
    timestamp = int(time.time())  # Utiliser l'heure actuelle en secondes
    return f"{base_name}-{timestamp}"
def get_aws_session():
    """
    Crée et retourne une session AWS.
    Charge les clés d'accès depuis un fichier .env pour une configuration sécurisée.
    """
    return boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY"),
        region_name="us-east-1"  # Vous pouvez changer la région si nécessaire
    )
def get_text_from_speech(filename, aws_service,job_name,bucket_name):
    """
    Convertit de la parole en texte en utilisant AWS Transcribe.

    Cette fonction téléverse un fichier audio spécifié dans un seau S3, démarre un travail de transcription avec AWS Transcribe,
    attend que le travail soit terminé, et récupère le texte transcrit.

    Paramètres :
    - filename (str) : Chemin local vers le fichier audio à transcrire.
    - aws_service (object) : Client AWS Transcribe configuré.
    - job_name (str) : Nom unique pour le travail de transcription.
    - bucket_name (str) : Nom du seau S3 où le fichier audio est stocké.

    Retourne :
    - str : Le texte transcrit du fichier audio.

    Prérequis :
    - Le fichier audio doit déjà être téléversé dans le seau S3 spécifié.
    """
    aws_session = get_aws_session()
    s3_client = aws_session.client('s3')

    # Téléverser le fichier audio dans S3
    try:
        s3_client.upload_file(filename, bucket_name, os.path.basename(filename))
        print(bucket_name)
        print(f"Fichier {filename} téléversé avec succès dans le seau {bucket_name}.")
    except Exception as e:
        print(f"Erreur lors du téléversement du fichier : {e}")
        return None

    # URI du fichier audio dans S3
    media_uri = f"s3://{bucket_name}/{os.path.basename(filename)}"

    # Créer un client AWS Transcribe
    transcribe_client = aws_session.client('transcribe', region_name='us-east-1')

    # Générer un nom unique pour la tâche de transcription
    unique_job_name = generate_unique_job_name(job_name)

    # Démarrer la tâche de transcription
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=unique_job_name,
            LanguageCode="fr-FR",  # Langue de la transcription (ici, français)
            Media={'MediaFileUri': media_uri},
            OutputBucketName=bucket_name  # Le bucket où stocker la sortie transcrite
        )
        print(f"Tâche de transcription '{unique_job_name}' lancée pour {filename}.")
    except Exception as e:
        print(f"Erreur lors du démarrage de la transcription : {e}")
        return None

    # Attendre la fin de la transcription
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=unique_job_name)
        job_status = status['TranscriptionJob']['TranscriptionJobStatus']
        
        if job_status in ['COMPLETED', 'FAILED']:
            break
        else:
            print("En attente de la fin de la transcription...")
            time.sleep(30)  # Attendre 30 secondes avant de vérifier à nouveau

    if job_status == 'COMPLETED':
        # Récupérer le résultat de la transcription
        transcription_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        print(f"Transcription terminée. Résultat disponible ici : {transcription_url}")

        # Télécharger le fichier JSON de transcription
        try:
            response = urllib.request.urlopen(transcription_url)
            transcript_data = json.loads(response.read())
            transcript_text = transcript_data['results']['transcripts'][0]['transcript']
            return transcript_text
        except Exception as e:
            print(f"Erreur lors du téléchargement ou de l'analyse du fichier de transcription : {e}")
            return None
    else:
        print(f"Le travail de transcription a échoué. Statut : {job_status}")
        return None





import os
import boto3
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Fonction pour créer la session AWS
def get_aws_session():
    """
    Crée et retourne une session AWS.
    Charge les clés d'accès depuis un fichier .env pour une configuration sécurisée.
    """
    return boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY"),
        region_name="us-east-1"  # Vous pouvez changer la région si nécessaire
    )

# Fonction pour extraire les expressions clés
def extract_keyphrases(text, aws_service):
    """
    Extrait les expressions clés d'un texte et retourne les 10 expressions les plus pertinentes comme hashtags.
    """
    try:
        # Appeler AWS Comprehend pour extraire les expressions clés
        response = aws_service.detect_key_phrases(Text=text, LanguageCode="fr")
        
        # Récupérer les expressions clés triées par score de pertinence
        key_phrases = [phrase['Text'] for phrase in sorted(response['KeyPhrases'], key=lambda x: x['Score'], reverse=True)]
        
        # Retourner les 10 expressions clés les plus pertinentes
        return key_phrases[:10]
    
    except Exception as e:
        print(f"Erreur lors de l'extraction des expressions clés : {e}")
        return []

# Créer la session AWS



import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Télécharger les stopwords si ce n'est pas déjà fait
nltk.download('stopwords')

def clean_text(raw_text):
    """
    Nettoie un texte en retirant les mots vides et en normalisant les mots en minuscules.

    Cette fonction prend un texte brut en entrée, tokenise le texte pour séparer les mots,
    convertit les mots en minuscules, et retire les mots vides (stop words) en français. Les mots vides
    supplémentaires peuvent être ajoutés à la liste. Le texte résultant contient uniquement les mots significatifs
    en minuscules.

    Paramètres :
    - raw_text (str) : Le texte brut à nettoyer.

    Retourne :
    - str : Le texte nettoyé, sans mots vides et en minuscules.

    Exemple d'utilisation :
    >>> texte_brut = "Ceci est un exemple de texte à nettoyer."
    >>> clean_text(texte_brut)
    'exemple texte nettoyer'
    """
    
    # Tokenizer pour séparer le texte en mots
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(raw_text.lower())  # Mettre en minuscules et tokeniser

    # Charger les stopwords en français
    stop_words = set(stopwords.words('french'))

    # Filtrer les mots pour retirer les mots vides et les mots trop courts (1-2 lettres)
    cleaned_text = [word for word in words if word not in stop_words and len(word) > 2]

    # Retourner le texte nettoyé sous forme d'une chaîne de caractères
    return ' '.join(cleaned_text)




import boto3

def extract_keyphrases(text, aws_service):
    """
    Extrait les expressions clés d'un texte et retourne les 10 expressions les plus pertinentes comme hashtags.

    Cette fonction utilise un service AWS, tel que Amazon Comprehend, pour détecter les expressions clés dans
    un texte donné. Elle trie ces expressions par leur score de pertinence fourni par AWS et retourne les 10
    expressions clés les plus pertinentes sous forme de hashtags.

    Paramètres :
    - text (str) : Le texte duquel extraire les expressions clés.
    - aws_service (object) : Un objet de service AWS configuré pour détecter les expressions clés.

    Retourne :
    - list[str] : Une liste des 10 hashtags les plus pertinents basés sur les expressions clés du texte.

    Exemple d'utilisation :
    >>> aws_comprehend_client = boto3.client('comprehend', region_name='us-east-1')
    >>> extract_keyphrases("Ceci est un exemple de texte.", aws_comprehend_client)
    ['#exemple', '#texte']
    """

    # Utilisation du service AWS Comprehend pour détecter les expressions clés
    try:
        # Appel à la méthode detect_key_phrases pour extraire les expressions clés
        response = aws_service.detect_key_phrases(Text=text, LanguageCode='fr')  # 'fr' pour français

        # Extraire les expressions clés avec leur score de confiance
        key_phrases = response['KeyPhrases']

        # Trier les expressions clés par score de pertinence (du plus élevé au plus bas)
        sorted_key_phrases = sorted(key_phrases, key=lambda x: x['Score'], reverse=True)

        # Utiliser un ensemble pour suivre les expressions déjà ajoutées
        added_phrases = set()

        # Créer la liste des hashtags sans doublons
        top_10_key_phrases = []
        for phrase in sorted_key_phrases:
            # Convertir chaque expression en hashtag
            hashtag = f"{phrase['Text'].replace(' ', '').lower()}"
            # Ajouter le hashtag si ce n'est pas déjà dans la liste
            if hashtag not in added_phrases:
                top_10_key_phrases.append(hashtag)
                added_phrases.add(hashtag)

            # Si on a déjà 10 hashtags, on arrête d'ajouter
            if len(top_10_key_phrases) == 10:
                break

        return top_10_key_phrases
    
    except Exception as e:
        print(f"Erreur lors de l'extraction des expressions clés : {e}")
        return []





import os
import boto3
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Fonction pour créer la session AWS
def get_aws_session():
    """
    Crée et retourne une session AWS.
    Charge les clés d'accès depuis un fichier .env pour une configuration sécurisée.
    """
    return boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY"),
        region_name="us-east-1"  # Vous pouvez changer la région si nécessaire
    )




def detect_objects(image_path, aws_service):
    """
    Détecte les objets dans une image en utilisant Amazon Rekognition.

    Cette fonction ouvre une image depuis un chemin spécifié, utilise un service AWS (Amazon Rekognition) pour
    détecter les objets présents dans l'image avec une confiance minimale de 50%, et retourne les noms des 10
    objets les plus pertinents détectés.

    Paramètres :
    - image_path (str) : Le chemin vers l'image à analyser.
    - aws_service (object) : Un client AWS Rekognition configuré.

    Retourne :
    - list[str] : Une liste contenant les noms des 10 premiers objets détectés dans l'image.

    Exemple d'utilisation :
    >>> aws_rekognition_client = boto3.client('rekognition', region_name='us-east-1')
    >>> detect_objects("/chemin/vers/image.jpg", aws_rekognition_client)
    ['Voiture', 'Arbre', 'Personne']
    """
    try:
        # Ouvrir l'image et la lire en binaire
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()

        # Appel à Amazon Rekognition pour détecter les objets
        response = aws_service.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=10,  # Limite à 10 objets détectés
            MinConfidence=50  # Confiance minimale de 50%
        )
        
        # Extraire les objets détectés avec leurs scores de confiance
        labels = response['Labels']
        
        # Trier les labels par score de confiance (du plus élevé au plus bas)
        sorted_labels = sorted(labels, key=lambda x: x['Confidence'], reverse=True)

        # Extraire les 10 objets avec les plus hauts scores de confiance
        top_objects = [label['Name'] for label in sorted_labels[:10]]

        return top_objects
    
    except Exception as e:
        print(f"Erreur lors de la détection des objets : {e}")
        return []







def detect_celebrities(image_path, aws_service):
    """
    Identifie les célébrités dans une image en utilisant le service Amazon Rekognition.

    Cette fonction ouvre une image depuis un chemin donné et utilise le service AWS Rekognition pour reconnaître les
    célébrités présentes dans l'image. Elle retourne une liste contenant les noms des célébrités identifiées, limitée
    aux 10 premiers résultats pour simplifier l'output.

    Paramètres :
    - image_path (str) : Le chemin vers l'image dans laquelle détecter les célébrités.
    - aws_service (object) : Un client AWS Rekognition configuré.

    Retourne :
    - list[str] : Une liste des noms des célébrités identifiées dans l'image, jusqu'à un maximum de 10.

    Exemple d'utilisation :
    >>> aws_rekognition_client = boto3.client('rekognition', region_name='us-east-1')
    >>> detect_celebrities("/chemin/vers/limage.jpg", aws_rekognition_client)
    ['Leonardo DiCaprio', 'Kate Winslet']
    """
    try:
        # Ouvrir l'image et la lire en binaire
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()

        # Appel à Amazon Rekognition pour détecter les célébrités
        response = aws_service.recognize_celebrities(
            Image={'Bytes': image_bytes}
        )

        # Extraire les célébrités détectées
        celebrities = response['CelebrityFaces']

        # Extraire les noms des célébrités détectées
        celebrity_names = [celebrity['Name'] for celebrity in celebrities]

        # Limiter à 10 célébrités
        top_celebrity_names = celebrity_names[:10]

        return top_celebrity_names

    except Exception as e:
        print(f"Erreur lors de la détection des célébrités : {e}")
        return []




def detect_emotions(image_path, aws_service):
    """
    Détecte les émotions sur les visages présents dans une image en utilisant Amazon Rekognition.
    
    Cette fonction analyse une image pour détecter les visages et leurs émotions associées.
    Pour chaque visage, elle retourne les émotions détectées avec leur niveau de confiance.
    
    Paramètres :
    - image_path (str) : Chemin vers l'image à analyser
    - aws_service (boto3.client) : Client AWS Rekognition configuré
    
    Retourne :
    - list[dict] : Liste des visages détectés avec leurs émotions
                  Format: [
                      {
                          'BoundingBox': dict,
                          'Emotions': [
                              {
                                  'Type': str,  # HAPPY, SAD, ANGRY, CONFUSED, etc.
                                  'Confidence': float
                              },
                              ...
                          ],
                          'AgeRange': {'Low': int, 'High': int},
                          'Gender': {'Value': str, 'Confidence': float}
                      },
                      ...
                  ]
    
    Exemple :
    >>> rekognition = boto3.client('rekognition')
    >>> emotions = detect_emotions("./photo.jpg", rekognition)
    >>> for face in emotions:
    ...     print(f"Émotions détectées : {face['Emotions']}")
    """
    try:
        # Ouvrir l'image et la lire en binaire
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()

        # Appel à Amazon Rekognition pour détecter les visages avec tous les attributs
        response = aws_service.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['ALL']  # Demander tous les attributs, y compris les émotions
        )

        # Initialiser la liste pour stocker les informations des visages détectés
        faces_info = []

        # Parcourir les visages détectés dans la réponse
        for face_detail in response['FaceDetails']:
            # Créer un dictionnaire pour chaque visage détecté
            face_data = {}

            # Récupérer le genre et son niveau de confiance
            face_data['Gender'] = {
                'Value': face_detail['Gender']['Value'],
                'Confidence': face_detail['Gender']['Confidence']
            }

            # Récupérer l'âge estimé (plage min-max)
            face_data['AgeRange'] = face_detail['AgeRange']

            # Récupérer les émotions avec leur niveau de confiance
            emotions = face_detail['Emotions']
            face_data['Emotions'] = sorted(emotions, key=lambda x: x['Confidence'], reverse=True)[:3]  # Prendre les 3 émotions principales

            # Ajouter les informations du visage à la liste
            faces_info.append(face_data)

            # Affichage des informations sur le visage détecté
            print("[INFO] Visage détecté:")
            print(f"  - Genre: {face_data['Gender']['Value']} (confiance: {face_data['Gender']['Confidence']}%)")
            print(f"  - Âge estimé: {face_data['AgeRange']['Low']}-{face_data['AgeRange']['High']} ans")
            print("  - Émotions principales:")
            for emotion in face_data['Emotions']:
                print(f"    * {emotion['Type']}: {emotion['Confidence']:.2f}%")
            print("---")

        return faces_info

    except Exception as e:
        print(f"Erreur lors de la détection des émotions : {e}")
        return []


def summarize_emotions(faces_info):
    """
    Résume les émotions détectées sur tous les visages d'une image.
    
    Cette fonction agrège les émotions de tous les visages et calcule les émotions
    dominantes dans l'image.
    
    Paramètres :
    - faces_info (list[dict]) : Liste des informations des visages détectés
    
    Retourne :
    - dict : Résumé des émotions dominantes et statistiques
    
    Exemple :
    >>> emotions = detect_emotions("./group_photo.jpg", rekognition)
    >>> summary = summarize_emotions(emotions)
    >>> print(f"Émotion dominante : {summary['dominant_emotion']}")
    """
    total_faces = len(faces_info)
    emotion_stats = {}
    age_stats = {'min': float('inf'), 'max': float('-inf'), 'total_age': 0, 'count': 0}
    gender_stats = {'Male': 0, 'Female': 0}
    dominant_emotion = {'Type': None, 'Confidence': 0}
    
    # Parcourir les visages détectés
    for face in faces_info:
        # Analyser le genre
        gender = face['Gender']['Value']
        gender_stats[gender] += 1
        
        # Analyser l'âge
        age_range = face['AgeRange']
        avg_age = (age_range['Low'] + age_range['High']) / 2
        age_stats['total_age'] += avg_age
        age_stats['count'] += 1
        age_stats['min'] = min(age_stats['min'], avg_age)
        age_stats['max'] = max(age_stats['max'], avg_age)
        
        # Analyser les émotions avec une confiance > 50%
        for emotion in face['Emotions']:
            if emotion['Confidence'] > 50:
                # Mettre à jour les statistiques des émotions
                emotion_type = emotion['Type']
                if emotion_type not in emotion_stats:
                    emotion_stats[emotion_type] = {'count': 0, 'total_confidence': 0}
                
                emotion_stats[emotion_type]['count'] += 1
                emotion_stats[emotion_type]['total_confidence'] += emotion['Confidence']
                
                # Mettre à jour l'émotion dominante
                if emotion['Confidence'] > dominant_emotion['Confidence']:
                    dominant_emotion['Type'] = emotion_type
                    dominant_emotion['Confidence'] = emotion['Confidence']
    
    # Calculer les moyennes des émotions
    for emotion_type, stats in emotion_stats.items():
        stats['average_confidence'] = stats['total_confidence'] / stats['count']
    
    # Calculer la moyenne des âges
    avg_age = age_stats['total_age'] / age_stats['count'] if age_stats['count'] > 0 else 0

    # Résumé final des résultats
    summary = {
        'total_faces': total_faces,
        'dominant_emotion': dominant_emotion['Type'],
        'dominant_emotion_confidence': dominant_emotion['Confidence'],
        'emotion_stats': emotion_stats,
        'age_stats': {
            'min_age': age_stats['min'],
            'max_age': age_stats['max'],
            'average_age': avg_age
        },
        'gender_stats': gender_stats
    }

    return summary



def print_face_details(faces_info):
    """
    Affiche les détails des visages détectés (genre, âge, émotions).
    """
    for idx, face in enumerate(faces_info):
        print(f"[INFO] Visage {idx + 1} détecté:")
        print(f"  - Genre: {face['Gender']['Value']} (confiance: {face['Gender']['Confidence']:.1f}%)")
        print(f"  - Âge estimé: {face['AgeRange']['Low']} - {face['AgeRange']['High']} ans")
        print(f"  - Émotions principales:")
        for emotion in face['Emotions']:
            print(f"    * {emotion['Type']}: {emotion['Confidence']:.1f}%")


def analyze_image(image_path):
    """
    Analyse une image et affiche les détails et statistiques des émotions.
    """
    # Détecter les émotions dans l'image
    faces_info = detect_emotions(image_path, aws_rekognition_client)
    
    # Afficher les détails de chaque visage détecté
    print(f"Analyser l'image : {image_path}")
    print_face_details(faces_info)
    
    # Résumer les émotions détectées
    summary = summarize_emotions(faces_info)
    
    # Afficher le résumé des statistiques
    print(f"\n[INFO] Résumé des émotions :")
    print(f"  - Nombre total de visages : {summary['total_faces']}")
    print(f"  - Émotion dominante : {summary['dominant_emotion']} (confiance: {summary['dominant_emotion_confidence']:.1f}%)")
    print(f"  - Statistiques des émotions :")
    for emotion, stats in summary['emotion_stats'].items():
        print(f"    * {emotion}: {stats['count']} occurrences, confiance moyenne: {stats['average_confidence']:.1f}%")
    
    print(f"  - Statistiques d'âge :")
    print(f"    * Âge min : {summary['age_stats']['min_age']} ans")
    print(f"    * Âge max : {summary['age_stats']['max_age']} ans")
    print(f"    * Âge moyen : {summary['age_stats']['average_age']:.1f} ans")
    
    print(f"  - Distribution des genres :")
    print(f"    * Hommes : {summary['gender_stats']['Male']}")
    print(f"    * Femmes : {summary['gender_stats']['Female']}")
    print("\n" + "-" * 50)


import os
import time
import cv2
import boto3
import tempfile
from matplotlib import pyplot as plt

def process_media(media_file, rekognition, transcribe, comprehend, bucket_name):
    """
    Traite un fichier multimédia (image ou vidéo) pour modérer le contenu, détecter des objets/célébrités,
    transcrire le discours et extraire des expressions clés.

    Selon le type de fichier, cette fonction applique une chaîne de traitement appropriée en utilisant différents
    services AWS. Pour les images, elle modère le contenu, détecte des objets, émotions faciales et des célébrités. Pour les vidéos,
    elle extrait une image, modère le contenu, téléverse la vidéo sur S3, transcrit le discours en texte, nettoie le texte,
    et extrait des expressions clés.

    Paramètres :
    - media_file (str) : Chemin vers le fichier multimédia à traiter.
    - rekognition (object) : Client AWS Rekognition configuré.
    - transcribe (object) : Client AWS Transcribe configuré.
    - comprehend (object) : Client AWS Comprehend configuré.
    - bucket_name (str) : Nom du seau S3 pour stocker les fichiers vidéo.

    Retourne :
    - dict : Dictionnaire contenant des hashtags pour les images ou des sous-titres et hashtags pour les vidéos.
    """
    file_extension = os.path.splitext(media_file)[1].lower()

    if file_extension in ['.jpg', '.jpeg', '.png']:
        # Si c'est une image, modérer l'image
        if moderate_image(media_file, rekognition) is not None:
            return None  # Contenu choquant détecté
        
        # Si l'image est modérée sans problème, détecter les objets, émotions et célébrités
        key_phrases = []

        # Détection des objets
        key_phrases.extend(detect_objects(media_file, rekognition))
        
        # Détection des émotions et des visages
        faces = detect_emotions(media_file, rekognition)  # Utilisation de votre fonction detect_emotions
        for face in faces:
            for emotion in face['Emotions']:
                if emotion['Confidence'] > 50:
                    key_phrases.append(f"{emotion['Type'].lower()}")
        
        # Détection des célébrités
        key_phrases.extend(detect_celebrities(media_file, rekognition))
        
        # Extraire des expressions clés (hashtags)
        text = ' '.join(key_phrases)
        # Utilisation de votre fonction extract_keyphrases
        
        # Retourner les hashtags
        return {'hashtags': list(set(key_phrases))}
    elif file_extension in ['.mp4', '.mov', '.avi']:
        frame1 = extract_frame_video(media_file, 1)
        
        if frame1 is not None:
            photo_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            plt.imshow(photo_rgb)
            plt.axis('off') 
            plt.title("Première image extraite")
            plt.show()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img_file:
                temp_img_path = temp_img_file.name
                cv2.imwrite(temp_img_path, frame1)  # Sauvegarder l'image en tant que fichier

            # Vérification de contenu choquant avec le fichier enregistré
            if moderate_image(temp_img_path, rekognition) is not None:
                return None  # Contenu choquant détecté

            os.remove(temp_img_path)
            job_name = 'transcriptionText'
            transcript_text = get_text_from_speech(media_file, transcribe, job_name,bucket_name)
            
            texte_nettoye = clean_text(transcript_text)
            key_phrases = extract_keyphrases(texte_nettoye, comprehend)
            return {'subtitles': transcript_text , 'hashtags': list(set(key_phrases))}
            
        
        
    return None
    