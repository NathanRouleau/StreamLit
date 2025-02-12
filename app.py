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

def load_env_credentials():
    load_dotenv(dotenv_path='.env') 
    access_key = os.getenv("ACCESS_KEY")
    secret_key = os.getenv("SECRET_KEY")
    bucket_name = os.getenv("BUCKET_NAME")
    return access_key, secret_key, bucket_name

def test_aws_credentials(access_key, secret_key, bucket_name):
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        s3_client.head_bucket(Bucket=bucket_name)
        st.sidebar.success("Connexion réussie au bucket S3!")
    except NoCredentialsError:
        st.sidebar.error("Erreur : Credentials AWS invalides.")
    except Exception as e:
        st.sidebar.error(f"Erreur de connexion : {str(e)}")

# Barre latérale pour la configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.subheader("🔑 Credentials AWS")

# Vérifie si les credentials ont été chargés depuis .env
if 'credentials_loaded' not in st.session_state:
    st.session_state.credentials_loaded = False

if st.sidebar.button("🔍 Charger les credentials depuis .env"):
    try:
        access_key, secret_key, bucket_name = load_env_credentials()
        if access_key and secret_key and bucket_name:
            st.session_state.access_key = access_key
            st.session_state.secret_key = secret_key
            st.session_state.bucket_name = bucket_name
        else:
            st.sidebar.error("Échec du chargement : certaines variables sont manquantes dans .env")
    except Exception as e:
        st.sidebar.error(f"Erreur lors du chargement des credentials : {e}")

# Champs pour modifier manuellement les credentials
access_key = st.sidebar.text_input("AWS Access Key ID", value=st.session_state.get("access_key", ""), type="password", disabled=st.session_state.credentials_loaded)
secret_key = st.sidebar.text_input("AWS Secret Access Key", value=st.session_state.get("secret_key", ""), type="password", disabled=st.session_state.credentials_loaded) 
bucket_name = st.sidebar.text_input("Nom du bucket S3", value=st.session_state.get("bucket_name", ""), disabled=st.session_state.credentials_loaded)

if access_key and secret_key and bucket_name:
    st.session_state.access_key = access_key
    st.session_state.secret_key = secret_key
    st.session_state.bucket_name = bucket_name
    st.sidebar.success("Credentials AWS enregistrés.")

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
        temp_file.write(uploaded_file.getbuffer())
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
            print(uploaded_file)
            result = process_media(temp_file_path, aws_rekognition_client, aws_transcribe_client, aws_comprehend_client, BUCKET_NAME)
            # Si contenu inapproprié
            if "error" in result: 
                st.warning(result["error"])
            if isinstance(result, list) or result is None:
                st.markdown(
                    """<div style="background-color: #3e2428; color: white; padding: 20px; text-align: center; border-radius: 10px; font-size: 24px;border-radius: 10px 10px 0 0;font-weight: bold;">
                        🚨 Contenu inapproprié détecté ! 🚨
                    </div>""", unsafe_allow_html=True
                )
                st.markdown(
                    """<div style="background-color: #3e2428; color: red; text-align: center;border-radius: 0 0 10px 10px; font-size: 20px;">
                        ❌ Cette publication a été bloquée
                    </div>""", unsafe_allow_html=True
                )
                
                # Liste des thèmes problématiques en cas d'image inappropriée 
                if result:
                    st.error("🔍 Thèmes détectés :")
                    for theme in result:
                        st.error(f"⚠️ **{theme}**")
            
            # Si l'image/video est correcte
            else:
                # Image
                if file_extension in ["jpg", "jpeg", "png"]:
                    st.image(temp_file_path, caption="Image validée ✅", use_container_width=True)

                    # Gestion des hashtags
                    hashtags = result.get("hashtags", [])
                    if hashtags:
                        st.markdown("### 🏷️ Hashtags générés :")
                        st.markdown("""<div style="display: flex; flex-wrap: wrap;">{}</div>"""
                            .format("".join([f'<span style="background-color: #b0f2b6; border-radius: 12px; padding: 5px 12px; margin: 5px; font-size: 14px; color: #333;">#{tag}</span>' for tag in hashtags])
                            ), unsafe_allow_html=True
                        )
                    else:
                        st.write("Aucun hashtag généré.")
                        st.success(f"Le fichier {uploaded_file.name} a été analysé avec succès.")

                # Vidéo
                if file_extension in ["mp4", "avi"]:
                    st.video(temp_file_path)

                    # Gestion des hashtags
                    hashtags = result.get("hashtags", [])
                    if hashtags:
                        st.markdown("### 🏷️ Hashtags générés :")
                        st.markdown("""<div style="display: flex; flex-wrap: wrap;">{}</div>"""
                            .format("".join([f'<span style="background-color: #b0f2b6; border-radius: 12px; padding: 5px 12px; margin: 5px; font-size: 14px; color: #333;">#{tag}</span>' for tag in hashtags])
                            ), unsafe_allow_html=True
                        )
                    else:
                        st.write("Aucun hashtag généré.")
                        st.success(f"Le fichier {uploaded_file.name} a été analysé avec succès.")

                    transcription = result.get("subtitles", "").strip()

                    if transcription:
                        st.markdown("📝 Transcription :")
                        st.text_area("Texte extrait de la vidéo :", transcription, height=250)
                    else:
                        st.write("Aucune transcription disponible.")

                    st.success(f"Le fichier {uploaded_file.name} a été analysé avec succès.")