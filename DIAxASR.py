import os
import string
import gc
import argparse
import tempfile
import subprocess
import shutil

import pandas as pd
import numpy as np

import pympi

from pydub import AudioSegment
from pydub.exceptions import CouldntEncodeError

import torch

from pyannote.audio import Pipeline
from pyannote.audio import Audio as PaAudio

from diarizers import SegmentationModel

from transformers import WhisperProcessor, WhisperForConditionalGeneration

from datasets import load_dataset, Audio, DatasetDict


# Fonctions utilitaires
def convert_sec(timecode):
    sec = int(timecode.split(".")[0]) * 1000 + int(timecode.split(".")[1])
    return sec

def convert_to_ms(timecode):
    hours, minutes, seconds = timecode.split(":")
    ms = int(hours) * 3600000 + int(minutes) * 60000 + int(seconds.split(".")[0]) * 1000 + int(seconds.split(".")[1])
    return ms

def get_ts(seg):
    ts = str(seg).replace('[', '').replace(']', '').replace(' ', '').split('-->')
    return ts


def convert_audio_files_to_wav(input_dir, tmp_dir):
    """
    Convertit tous les fichiers audio (mp3, m4a, etc.) en fichiers .wav (mono, 16kHz)
    et les place dans un dossier temporaire.

    Args:
        input_dir (str): Dossier d'entrée avec fichiers audio.
        tmp_dir (str): Dossier temporaire pour stocker les fichiers convertis.

    Returns:
        str: Le chemin du dossier contenant les fichiers wav convertis.
    """

    os.makedirs(tmp_dir, exist_ok=True)
    supported_formats = (".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma", ".wav")
    converted = 0

    for filename in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, filename)
        base, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext not in supported_formats:
            continue

        try:
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            wav_path = os.path.join(tmp_dir, f"{base}.wav")
            audio.export(wav_path, format="wav")
            print(f"✅ Converti : {filename} -> {wav_path}")
            converted += 1
        except Exception as e:
            print(f"⚠️ Erreur de conversion pour {filename} : {e}")

    print(f"🎧 Conversion terminée. {converted} fichiers convertis dans {tmp_dir}.")
    return tmp_dir

def clean_temp_wavs(tmp_dir):
    """
    Supprime le dossier temporaire contenant les fichiers .wav convertis.

    Args:
        tmp_dir (str): Chemin vers le dossier temporaire à supprimer.
    """
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"🧹 Dossier temporaire supprimé : {tmp_dir}")

# Chargement du pipeline de diarisation Pyannote avec modèle custom
def load_diarization_pipeline(model_id, hf_token, device):
    device = torch.device(device)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token=hf_token).to(device)

    if model_id:
        model = SegmentationModel().from_pretrained(model_id)
        model = model.to_pyannote_model()
        pipeline._segmentation.model = model.to(device)
    else:
        model_id = "pyannote/segmentation-3.0"

    print(f"Pipeline de diarisation chargé avec le modèle : {model_id} sur {device}")
    return pipeline


# Diarisation de tous les fichiers WAV d'un dossier avec un pipeline donné
def run_diarization_on_dir(wav_dir, output_dir, pipeline, output_format="tsv"):
    assert output_format in ["eaf", "tsv"], "output_format must be either 'eaf' or 'tsv'"

    io = PaAudio(mono='downmix', sample_rate=16000)
    output_subdir = os.path.join(output_dir, output_format)
    os.makedirs(output_subdir, exist_ok=True)

    for wav_file in sorted(os.listdir(wav_dir)):
        if not wav_file.endswith(".wav"):
            print(f"Le fichier {wav_file} n'est pas au format wav")
            continue

        full_path = os.path.join(wav_dir, wav_file)
        basename = os.path.splitext(wav_file)[0]
        waveform, sample_rate = io(full_path)
        # diarization_result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
        diarization_result = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        diarization_result_temp = [i.split(' ') for i in str(diarization_result).split('\n') if i.strip() != '']
        locuteurs = list(set([i[6] for i in diarization_result_temp]))

        if output_format == 'eaf':
            eaf = pympi.Elan.Eaf()
            eaf.add_linked_file(full_path)
            for locuteur in locuteurs:
                eaf.add_tier(locuteur)
        else:
            annotations = []
            audio = AudioSegment.from_wav(full_path)
            segment_dir = os.path.join(output_dir, f"{basename}_segments")
            os.makedirs(segment_dir, exist_ok=True)

        for seg, _, spk in diarization_result.itertracks(yield_label=True):
            ts_begin = int(seg.start * 1000)
            ts_end = int(seg.end * 1000)
            text = ""
            keep_punct = ["?", "'"]
            text = "".join([char for char in text if char not in string.punctuation or char in keep_punct])

            if not spk:
                spk = locuteurs[0]

            if ts_begin < ts_end:
                if output_format == 'eaf':
                    eaf.add_annotation(spk, ts_begin, ts_end, text)
                else:
                    extract = audio[ts_begin:ts_end]
                    audio_filename = os.path.join(segment_dir, f"{basename}_{spk}_{ts_begin}_{ts_end}.wav")
                    extract.export(audio_filename, format="wav")
                    timecodes = f"[{ts_begin}, {ts_end}]"
                    annotations.append([audio_filename, text, timecodes, spk])

        if output_format == 'eaf':
            out_file = os.path.join(output_subdir, f"{basename}.eaf")
            eaf.to_file(out_file)
        else:
            df = pd.DataFrame(annotations, columns=["audio", "text", "timecodes", "speaker"])
            out_file = os.path.join(output_subdir, f"{basename}.tsv")
            df.to_csv(out_file, sep='\t', index=False)

        print(f"Fichier exporté : {out_file}")


# Construction du dataset pour le traitement ASR à partir de l'EAF
def build_dataset_from_eaf_dir(eaf_dir, wav_dir, output_base_dir):
    os.makedirs(os.path.join(output_base_dir, "tsv"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "wav"), exist_ok=True)
    annotations_passed = 0
    data = []

    for file_name in os.listdir(eaf_dir):
        if file_name.endswith(".eaf"):
            base_name = file_name[:-4]
            eaf_path = os.path.join(eaf_dir, file_name)
            wav_path = os.path.join(wav_dir, base_name + ".wav")

            if not os.path.exists(wav_path):
                print(f"⚠️ Fichier WAV manquant pour : {base_name}")
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", wav_path,
                    "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                    tmp_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Erreur ffmpeg lors de la conversion de {wav_path} : {e}")
                continue

            eaf = pympi.Elan.Eaf(eaf_path)
            try:
                audio = AudioSegment.from_file(tmp_path, format="wav")
            except Exception as e:
                print(f"⚠️ Erreur chargement audio converti {tmp_path} : {e}")
                os.remove(tmp_path)
                continue

            for tier_name in eaf.tiers:
                if "_extra" not in tier_name:
                    for annotation in eaf.get_annotation_data_for_tier(tier_name):
                        start_time, end_time, value = annotation
                        duration = (end_time - start_time) / 1000
                        value = value.strip()

                        if duration > 30 or (value.startswith("[") and value.endswith("]")):
                            annotations_passed += 1
                            continue

                        extract = audio[start_time:end_time]

                        if len(extract) == 0 or extract.frame_count() == 0:
                            print(f"⚠️ Segment vide ignoré : {base_name} {tier_name} {start_time}-{end_time}")
                            annotations_passed += 1
                            continue

                        output_wav = os.path.join(output_base_dir, "wav", f"{base_name}_{tier_name}_{start_time}_{end_time}.wav")

                        try:
                            extract.export(output_wav, format="wav")
                        except Exception as e:
                            print(f"⚠️ Export échoué pour {output_wav} : {e}")
                            annotations_passed += 1
                            continue

                        timecodes = f"[{start_time}, {end_time}]"
                        data.append([output_wav, value, timecodes, tier_name])

            os.remove(tmp_path)

    df = pd.DataFrame(data, columns=["audio", "text", "timecodes", "speaker"])
    for base_name in df['audio'].apply(lambda x: os.path.basename(x).split('_')[0]).unique():
        df_part = df[df['audio'].str.contains(base_name)]
        output_tsv_path = os.path.join(output_base_dir, "tsv", f"{base_name}.tsv")
        df_part.to_csv(output_tsv_path, sep='\t', index=False)
        print(f"✅ TSV généré : {output_tsv_path}")

    print(f"⚠️ Annotations ignorées : {annotations_passed}")
    return os.path.join(output_base_dir, "tsv")

# Chargement des modèles Whisper
def load_whisper_model(model_id, device):
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model = model.to(device)
    print(f"Modèle Whisper chargé : {model_id} sur {device}")
    return processor, model


# Chargement des datasets audio à partir d'un dossier TSV
def load_dataset_from_tsv_dir(tsv_dir):
    dataset_dict = {}
    for file_name in sorted(os.listdir(tsv_dir)):
        if file_name.endswith(".tsv"):
            base_name = file_name[:-4]
            file_path = os.path.join(tsv_dir, file_name)

            dataset = load_dataset(
                'csv',
                data_files={"default": file_path},
                delimiter='\t',
                column_names=['audio', 'text', 'timecodes', 'speaker'],
                skiprows=1)['default'].cast_column("audio", Audio(sampling_rate=16000))

            dataset_dict[base_name] = dataset
    return dataset_dict


# Transcription des datasets et sauvegarde des résultats dans un dossier TSV
def transcribe_dataset_dir(dataset_dict, processor, model, output_dir, language):
    os.makedirs(output_dir, exist_ok=True)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    for base_name, dataset in dataset_dict.items():
        updated_data = []

        for item in dataset:
            audio = AudioSegment.from_file(item["audio"]["path"], format="wav")

            if len(audio) < 1000:
                padding = AudioSegment.silent(duration=1000 - len(audio), frame_rate=16000)
                audio += padding

            samples = np.array(audio.set_frame_rate(16000).set_sample_width(2).set_channels(1).get_array_of_samples()).astype(np.float32) / 32768.0

            if len(samples) < 480000:
                padded = np.zeros(480000, dtype=np.float32)
                padded[:len(samples)] = samples
            else:
                padded = samples[:480000]

            input_dict = processor(
                padded,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(model.device)

            predicted_ids = model.generate(input_dict, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            updated_data.append({
                "audio": item["audio"]["path"],
                "text": transcription,
                "timecodes": item["timecodes"],
                "speaker": item["speaker"]
            })

        df = pd.DataFrame(updated_data)
        out_path = os.path.join(output_dir, f"{base_name}.tsv")
        df.to_csv(out_path, sep='\t', index=False)
        print(f"Transcription enregistrée : {out_path}")


# Mise à jour des fichiers EAF avec les transcriptions associées
def update_eaf_with_tsv_dir(eaf_dir, tsv_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(tsv_dir):
        if file_name.endswith(".tsv"):
            base_name = file_name[:-4]
            tsv_path = os.path.join(tsv_dir, file_name)
            eaf_path = os.path.join(eaf_dir, f"{base_name}.eaf")

            if not os.path.exists(eaf_path):
                print(f"Fichier EAF manquant pour : {base_name}")
                continue

            eaf = pympi.Elan.Eaf(eaf_path)
            df = pd.read_csv(tsv_path, sep='\t')

            for _, row in df.iterrows():
                speaker = row['speaker']
                text = row['text']
                timecodes = eval(row['timecodes'])
                start_time, end_time = int(timecodes[0]), int(timecodes[1])

                if speaker in eaf.tiers:
                    for ann in eaf.get_annotation_data_for_tier(speaker):
                        ann_start, ann_end, ann_value = ann
                        if ann_value == "" and ann_start == start_time and ann_end == end_time:
                            eaf.remove_annotation(speaker, ann_start, ann_end)
                            eaf.add_annotation(speaker, ann_start, ann_end, value=text)
                        elif ann_value.strip(" ") == "!" and ann_start == start_time and ann_end == end_time:
                            eaf.remove_annotation(speaker, ann_start, ann_end)
                            eaf.add_annotation(speaker, ann_start, ann_end, value=text)    
                            # print(f"Annotation mise à jour : {speaker}, {start_time}-{end_time}, {text}")

            output_eaf_path = os.path.join(output_dir, f"{base_name}.eaf")
            eaf.to_file(output_eaf_path)
            print(f"Fichier EAF mis à jour : {output_eaf_path}")

def eaf2textgrid(eaf_path, textgrid_path):
    eaf = pympi.Elan.Eaf(eaf_path)
    textgrid = eaf.to_textgrid()
    textgrid.to_file(textgrid_path)

def textgrid2eaf(textgrid_path, eaf_path):
    textgrid = pympi.Praat.TextGrid(textgrid_path)
    eaf = textgrid.to_eaf(skipempty=True)
    eaf.to_file(eaf_path)

# Fonction pour appel depuis une interface Gradio
def run_pipeline_from_interface(
    wav_dir,           # dossier audio
    input_dir,         # dossier EAF ou TextGrid à transcrire (optionnel)
    output_dir,        # dossier de sortie
    diar_model_id,     # modèle pyannote
    asr_model_id,      # modèle Whisper
    hf_token,          # token HF
    output_format,     # "eaf", "tsv" ou "textgrid"
    language,          # langue ASR
    device,            # "cpu" ou "cuda"
    mode='pipeline',   # "diarize", "transcribe", "pipeline"
    keep_temp_audio=False
):
    """
    Interface web pour le pipeline DIAxASR multi-format.
    """
    torch.cuda.empty_cache()
    gc.collect()
    
    # Conversion des audios en wav
    tmp_wav_dir = os.path.join(output_dir, "_temp_wavs")
    converted_wav_dir = convert_audio_files_to_wav(wav_dir, tmp_wav_dir)
    
    eaf_dir_for_asr = None
    temp_eaf_dir = None

    try:
        # DIARISATION
        if mode in ['diarize', 'pipeline']:
            diar_out_format = 'eaf' if output_format == 'textgrid' else output_format
            pipeline = load_diarization_pipeline(diar_model_id, hf_token, device)
            run_diarization_on_dir(converted_wav_dir, output_dir, pipeline, diar_out_format)
            if mode == 'pipeline':
                eaf_dir_for_asr = os.path.join(output_dir, "eaf")

        # PRÉPARATION DES FICHIERS SEGMENTS POUR ASR
        if (mode == 'transcribe') or (mode == 'pipeline' and input_dir):
            if not input_dir:
                raise ValueError("input_dir (EAF ou TextGrid) requis pour la transcription.")
            entries = [f for f in os.listdir(input_dir) if not f.startswith('.')]
            if all(f.lower().endswith('.eaf') for f in entries):
                eaf_dir_for_asr = input_dir
            elif all(f.lower().endswith('.textgrid') for f in entries):
                temp_eaf_dir = os.path.join(output_dir, "_temp_eaf_from_textgrid")
                os.makedirs(temp_eaf_dir, exist_ok=True)
                for fname in entries:
                    if fname.lower().endswith(".textgrid"):
                        textgrid_path = os.path.join(input_dir, fname)
                        eaf_path = os.path.join(temp_eaf_dir, fname.replace(".TextGrid", ".eaf").replace(".textgrid", ".eaf"))
                        textgrid2eaf(textgrid_path, eaf_path)
                        print(f"✅ Converti {textgrid_path} -> {eaf_path}")
                eaf_dir_for_asr = temp_eaf_dir
            else:
                raise ValueError("input_dir doit contenir uniquement des .eaf ou .TextGrid")
        
        tsv_dir = None
        if (mode == 'transcribe') or (mode == 'pipeline' and eaf_dir_for_asr):
            tsv_dir = build_dataset_from_eaf_dir(eaf_dir_for_asr, converted_wav_dir, output_dir)
        elif mode == 'pipeline' and not eaf_dir_for_asr:
            eaf_dir_for_asr = os.path.join(output_dir, "eaf")
            if not os.path.exists(eaf_dir_for_asr):
                raise FileNotFoundError(f"Dossier EAF non trouvé à {eaf_dir_for_asr}.")
            tsv_dir = build_dataset_from_eaf_dir(eaf_dir_for_asr, converted_wav_dir, output_dir)
        else:
            tsv_dir = os.path.join(output_dir, "tsv")

        # ASR
        if mode in ['transcribe', 'pipeline']:
            processor, model = load_whisper_model(asr_model_id, device)
            if not os.path.exists(tsv_dir):
                raise FileNotFoundError(f"Le dossier TSV {tsv_dir} n'existe pas.")
            dataset_dict = load_dataset_from_tsv_dir(tsv_dir)
            transcribed_tsv_dir = os.path.join(output_dir, "tsv_transcribed")
            transcribe_dataset_dir(dataset_dict, processor, model, transcribed_tsv_dir, language=language)

            # Mise à jour EAF avec transcriptions (si besoin)
            if output_format in ["eaf", "textgrid"]:
                eaf_updated_dir = os.path.join(output_dir, "eaf_updated")
                update_eaf_with_tsv_dir(eaf_dir_for_asr, transcribed_tsv_dir, eaf_updated_dir)
        
        # Conversion EAF→TextGrid (si demandé)
        if output_format == 'textgrid':
            eaf_dir_to_convert = os.path.join(output_dir, "eaf_updated") \
                if mode in ['transcribe', 'pipeline'] else os.path.join(output_dir, "eaf")
            textgrid_dir = os.path.join(output_dir, "textgrid_updated")
            os.makedirs(textgrid_dir, exist_ok=True)
            for fname in os.listdir(eaf_dir_to_convert):
                if fname.endswith(".eaf"):
                    eaf_path = os.path.join(eaf_dir_to_convert, fname)
                    tg_path = os.path.join(textgrid_dir, fname.replace(".eaf", ".TextGrid"))
                    eaf2textgrid(eaf_path, tg_path)
                    print(f"✅ Converti {eaf_path} -> {tg_path}")
    
    finally:
        if not keep_temp_audio:
            clean_temp_wavs(tmp_wav_dir)
        # if temp_eaf_dir and os.path.exists(temp_eaf_dir):
        #     shutil.rmtree(temp_eaf_dir)

    return f"✅ Pipeline terminé. Résultats disponibles dans : {output_dir}"

# CLI principale
def main():
    torch.cuda.empty_cache()
    gc.collect()

    parser = argparse.ArgumentParser(description="Pipeline de diarisation et transcription avec conversion audio.")
    parser.add_argument('--mode', type=str, required=True, choices=['diarize', 'transcribe', 'pipeline'],
                        help="Mode d'exécution : diarize, transcribe, pipeline")
    parser.add_argument('--wav-dir', type=str, required=True, help="Dossier contenant les fichiers audio à traiter")
    parser.add_argument('--input-dir', type=str, default=None, help="Dossier contenant les fichiers EAF ou TextGrid (entrée manuelle ou issue de la diarisation)")
    parser.add_argument('--output-dir', type=str, required=True, help="Dossier de sortie pour les résultats")
    parser.add_argument('--segmentation-model-id', type=str, required=False, help="ID du modèle Pyannote (segmentation)")
    parser.add_argument('--asr-model-id', type=str, required=False, default="openai/whisper-large-v3-turbo",
                        help="ID du modèle Whisper pour l'ASR")
    parser.add_argument('--hf-token', type=str, required=False, help="Hugging Face token pour Pyannote")
    parser.add_argument('--out-format', type=str, choices=['eaf', 'tsv', 'textgrid'], default='tsv',
                        help="Format de sortie (eaf, tsv ou textgrid)")
    parser.add_argument('--language', type=str, required=False, help="Code langue pour Whisper (ex : fr, ht, en)")
    parser.add_argument('--keep-temp-audio', action='store_true',
                        help="Si spécifié, conserve les fichiers audio temporaires après exécution.")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                        help="Choisir le device pour l'inférence : 'cpu' ou 'cuda'. Par défaut, utilise cuda si disponible.")

    args = parser.parse_args()

    # Choix du device
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    tmp_wav_dir = os.path.join(args.output_dir, "_temp_wavs")
    converted_wav_dir = convert_audio_files_to_wav(args.wav_dir, tmp_wav_dir)

    # Initialisation du dossier EAF à utiliser pour l'ASR (input réel ou conversion à la volée)
    eaf_dir_for_asr = None
    temp_eaf_dir = None

    try:
        # Étape de diarisation
        if args.mode in ['diarize', 'pipeline']:
            diar_out_format = 'eaf' if args.out_format == 'textgrid' else args.out_format
            pipeline = load_diarization_pipeline(args.segmentation_model_id, args.hf_token, device)
            run_diarization_on_dir(converted_wav_dir, args.output_dir, pipeline, diar_out_format)
            # Si pipeline, eaf_dir_for_asr = dossier eaf généré par la diarisation
            if args.mode == 'pipeline':
                eaf_dir_for_asr = os.path.join(args.output_dir, "eaf")

        # Préparation de l'entrée pour ASR (transcribe ou pipeline)
        if args.mode == 'transcribe' or (args.mode == 'pipeline' and args.input_dir):
            if not args.input_dir:
                raise ValueError("--input-dir est requis pour la transcription.")
            # Vérifie le contenu du dossier input_dir : eaf ou textgrid
            entries = [f for f in os.listdir(args.input_dir) if not f.startswith('.')]
            if all(f.lower().endswith('.eaf') for f in entries):
                eaf_dir_for_asr = args.input_dir
            elif all(f.lower().endswith('.textgrid') for f in entries):
                # Conversion TextGrid → EAF
                temp_eaf_dir = os.path.join(args.output_dir, "_temp_eaf_from_textgrid")
                os.makedirs(temp_eaf_dir, exist_ok=True)
                for fname in entries:
                    if fname.lower().endswith(".textgrid"):
                        textgrid_path = os.path.join(args.input_dir, fname)
                        eaf_path = os.path.join(temp_eaf_dir, fname.replace(".TextGrid", ".eaf").replace(".textgrid", ".eaf"))
                        textgrid2eaf(textgrid_path, eaf_path)
                        print(f"✅ Converti {textgrid_path} -> {eaf_path}")
                eaf_dir_for_asr = temp_eaf_dir
            else:
                raise ValueError("Le dossier --input-dir doit contenir soit uniquement des .eaf, soit uniquement des .TextGrid.")

        # Extraction des segments pour ASR (transcribe/pipeline)
        tsv_dir = None
        if args.mode == 'transcribe' or (args.mode == 'pipeline' and eaf_dir_for_asr):
            tsv_dir = build_dataset_from_eaf_dir(eaf_dir_for_asr, converted_wav_dir, args.output_dir)
        elif args.mode == 'pipeline' and not eaf_dir_for_asr:
            # En pipeline, si pas d'input_dir fourni, on suppose que les EAF viennent d'avant
            eaf_dir_for_asr = os.path.join(args.output_dir, "eaf")
            if not os.path.exists(eaf_dir_for_asr):
                raise FileNotFoundError(f"Dossier EAF non trouvé à {eaf_dir_for_asr}.")
            tsv_dir = build_dataset_from_eaf_dir(eaf_dir_for_asr, converted_wav_dir, args.output_dir)
        else:
            tsv_dir = os.path.join(args.output_dir, "tsv")

        # Étape ASR (sur les segments)
        if args.mode in ['transcribe', 'pipeline']:
            processor, model = load_whisper_model(args.asr_model_id, device)
            if not os.path.exists(tsv_dir):
                raise FileNotFoundError(f"Le dossier TSV {tsv_dir} n'existe pas. Vérifiez l'étape précédente.")
            dataset_dict = load_dataset_from_tsv_dir(tsv_dir)
            transcribed_tsv_dir = os.path.join(args.output_dir, "tsv_transcribed")
            transcribe_dataset_dir(dataset_dict, processor, model, transcribed_tsv_dir, language=args.language)

            # Mise à jour des EAF avec transcriptions
            if args.out_format in ["eaf", "textgrid"]:
                eaf_updated_dir = os.path.join(args.output_dir, "eaf_updated")
                update_eaf_with_tsv_dir(eaf_dir_for_asr, transcribed_tsv_dir, eaf_updated_dir)

        # Conversion EAF -> TextGrid si demandé
        if args.out_format == 'textgrid':
            eaf_dir_to_convert = os.path.join(args.output_dir, "eaf_updated") \
                if args.mode in ['transcribe', 'pipeline'] else os.path.join(args.output_dir, "eaf")
            textgrid_dir = os.path.join(args.output_dir, "textgrid_updated")
            os.makedirs(textgrid_dir, exist_ok=True)
            for fname in os.listdir(eaf_dir_to_convert):
                if fname.endswith(".eaf"):
                    eaf_path = os.path.join(eaf_dir_to_convert, fname)
                    tg_path = os.path.join(textgrid_dir, fname.replace(".eaf", ".TextGrid"))
                    eaf2textgrid(eaf_path, tg_path)
                    print(f"✅ Converti {eaf_path} -> {tg_path}")

    finally:
        if not args.keep_temp_audio:
            clean_temp_wavs(tmp_wav_dir)

        # if temp_eaf_dir and os.path.exists(temp_eaf_dir):
        #     shutil.rmtree(temp_eaf_dir)

if __name__ == "__main__":
    main()
