import tempfile
import time
import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import os
from moderation import *
import tempfile

aws_session = get_aws_session()
aws_rekognition_client = aws_session.client('rekognition', region_name='us-east-1')
aws_comprehend_client = aws_session.client('comprehend', region_name='us-east-1')
aws_transcribe_client = aws_session.client('transcribe', region_name='us-east-1')
BUCKET_NAME = 's3-transcriptionspeachtest'

# Fonction pour charger les credentials depuis le fichier .env
def load_credentials_from_env(file):
    with open("temp.env", "wb") as f:
        f.write(file.getbuffer())
    load_dotenv("temp.env")
    access_key = os.getenv("ACCESS_KEY")
    secret_key = os.getenv("SECRET_KEY")
    bucket_name = os.getenv("BUCKET_NAME")
    return access_key, secret_key, bucket_name

# Fonction pour tester les credentials AWS
def test_aws_credentials(access_key, secret_key, bucket_name):
    try:
        s3_client = boto3.client(
            "s3",
            access_key_id=access_key,
            secret_access_key=secret_key
        )
        s3_client.head_bucket(Bucket=bucket_name)
        st.sidebar.success("Connexion réussie au bucket S3!")
    except NoCredentialsError:
        st.sidebar.error("Erreur : Credentials AWS invalides.")
    except Exception as e:
        st.sidebar.error(f"Erreur de connexion : {str(e)}")

# Barre latérale pour la configuration
st.sidebar.title("Configuration")
st.sidebar.subheader("Credentials AWS")

# Option de chargement du fichier .env
env_file = st.sidebar.file_uploader("Charger credentials depuis .env", type=["env"])

# Méthode automatique de chargement des credentials
if env_file is not None:
    try:
        access_key, secret_key, bucket_name = load_credentials_from_env(env_file)
        st.sidebar.text_input("AWS Access Key ID", value=access_key, key="access_key", disabled=True)
        st.sidebar.text_input("AWS Secret Access Key", value=secret_key, key="secret_key", disabled=True)
        st.sidebar.text_input("Nom du bucket S3", value=bucket_name, key="bucket_name", disabled=True)
        st.sidebar.success("Credentials chargés depuis le fichier .env avec succès!")
    except Exception as e:
        st.sidebar.error(f"Erreur lors du chargement des credentials depuis .env : {e}")

# Méthode manuelle de saisie des credentials
access_key = st.sidebar.text_input("AWS Access Key ID", value="", type="password")
secret_key = st.sidebar.text_input("AWS Secret Access Key", value="", type="password")
bucket_name = st.sidebar.text_input("Nom du bucket S3", value="")

if access_key and secret_key and bucket_name:
    st.sidebar.success("Credentials AWS saisis manuellement.")

# Bouton pour tester la connexion
if st.sidebar.button("Tester les credentials AWS"):
    if access_key and secret_key and bucket_name:
        test_aws_credentials(access_key, secret_key, bucket_name)
    else:
        st.sidebar.error("Veuillez remplir tous les champs de credentials.")

# Sauvegarder les credentials dans l'état de session
if access_key and secret_key and bucket_name:
    st.session_state.access_key = access_key
    st.session_state.secret_key = secret_key
    st.session_state.bucket_name = bucket_name


# Taille maximale en octets (par exemple 10 MB)
MAX_SIZE = 10 * 1024 * 1024  # 10 MB

# Zone d'upload
st.title("Upload de contenu")
st.subheader("Choisissez un fichier (image ou vidéo)")

# Zone d'upload
uploaded_file = st.file_uploader("Glissez-déposez un fichier ou cliquez pour ouvrir l'explorateur", type=["jpg", "jpeg", "png", "mp4", "avi"])

# Si un fichier est sélectionné
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(uploaded_file.getbuffer())  # Sauvegarde du fichier
        temp_file_path = temp_file.name
    st.write("Fichier sélectionné:", uploaded_file.name)

    # Vérifier le format du fichier
    allowed_extensions = ["jpg", "jpeg", "png", "mp4", "avi"]

    if file_extension not in allowed_extensions:
        st.error("Format de fichier non autorisé. Seuls les fichiers image (jpg, jpeg, png) et vidéo (mp4, avi) sont autorisés.")
    # Vérification de la taille
    elif uploaded_file.size > MAX_SIZE:
        st.error(f"La taille du fichier dépasse la limite de {MAX_SIZE / (1024 * 1024)} MB.")
    else:
        with st.spinner("Analyse en cours..."):
            # Simuler un traitement du fichier (remplacer par ton code de traitement réel)
            #time.sleep(2)  
            print(uploaded_file)
            result = process_media(temp_file_path, aws_rekognition_client, aws_transcribe_client, aws_comprehend_client, BUCKET_NAME)
            if result is None : 
                st.error("🚨 Contenu inapproprié détecté ! 🚨")
                st.error("❌ Cette publication a été bloquée") 
                # st.error("🔍 Thèmes détectés :")
                # for theme in themes_detectes:
                #     st.markdown(f"- ❌ **{theme}**", unsafe_allow_html=True)

            else:
                if file_extension in ["jpg", "jpeg", "png"]:
                    st.image(temp_file_path, caption="Image validée ✅", use_container_width=True)

                    hashtags = result.get("hashtags", [])
                    if hashtags:
                        st.markdown("### 🏷️ Hashtags générés :")
                        st.markdown(", ".join([f"**#{tag}**" for tag in hashtags]))
                    else:
                        st.write("Aucun hashtag généré.")
                        st.success(f"Le fichier {uploaded_file.name} a été analysé avec succès.")


            # Sauvegarder le fichier temporairement
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                st.success(f"Le fichier a été sauvegardé temporairement sous : {temp_file.name}")


# # Fonction pour afficher le contenu approprié
# def display_appropriate_content(content_type, uploaded_file, hashtags):
#     st.subheader("Contenu approprié")

#     if content_type == "image":
#         # Affichage de l'image
#         st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
#     elif content_type == "video":
#         # Affichage de la vidéo
#         st.video(uploaded_file, caption="Vidéo téléchargée")

#     # Affichage des hashtags
#     st.write("Hashtags générés :")
#     for hashtag in hashtags:
#         st.markdown(f"<span style='color: #6c63ff; font-weight: bold;'>{hashtag}</span>", unsafe_allow_html=True)

#     # Option de transcription pour la vidéo (si vidéo)
#     if content_type == "video":
#         st.write("Transcription vidéo :")
#         # Ici tu pourrais ajouter une fonction pour la transcription automatique via un service externe
#         st.text("Transcription à venir...")

# # Fonction pour afficher une alerte en cas de contenu inapproprié
# def display_inappropriate_content(sensitive_themes):
#     st.markdown("<div style='background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 5px;'>", unsafe_allow_html=True)
#     st.markdown("<h3>Attention !</h3>", unsafe_allow_html=True)
#     st.markdown("Le contenu a été bloqué pour les raisons suivantes : ", unsafe_allow_html=True)

#     # Liste des thèmes sensibles détectés
#     for theme in sensitive_themes:
#         st.markdown(f"- {theme}", unsafe_allow_html=True)
    
#     st.markdown("</div>", unsafe_allow_html=True)

# # Logique pour afficher le contenu en fonction de son appropriateness
# if is_appropriate:
#     display_appropriate_content(content_type, uploaded_file, hashtags)
# else:
#     display_inappropriate_content(sensitive_themes)