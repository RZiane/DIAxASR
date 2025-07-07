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


# Chargement du pipeline de diarisation Pyannote avec modèle custom
def load_diarization_pipeline(model_id, hf_token):
    import torch
    from pyannote.audio import Pipeline
    from diarizers import SegmentationModel

    device = torch.device("cuda")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token=hf_token).to(device)

    if model_id:
        model = SegmentationModel().from_pretrained(model_id)
        model = model.to_pyannote_model()
        pipeline._segmentation.model = model.to(device)
    
    else:
        model_id = "pyannote/segmentation-3.0"

    print(f"Pipeline de diarisation chargé avec le modèle : {model_id}")
    return pipeline


# Diarisation de tous les fichiers WAV d'un dossier avec un pipeline donné
def run_diarization_on_dir(wav_dir, output_dir, pipeline, output_format="tsv"):
    import os
    import string
    import pandas as pd
    from pydub import AudioSegment
    from pyannote.audio import Audio as PaAudio
    import pympi

    assert output_format in ["eaf", "tsv"], "output_format must be either 'eaf' or 'tsv'"

    io = PaAudio(mono='downmix', sample_rate=16000)
    output_subdir = os.path.join(output_dir, output_format)
    os.makedirs(output_subdir, exist_ok=True)

    for wav_file in sorted(os.listdir(wav_dir)):
        if not wav_file.endswith(".wav"):
            continue

        full_path = os.path.join(wav_dir, wav_file)
        basename = os.path.splitext(wav_file)[0]
        waveform, sample_rate = io(full_path)
        diarization_result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)

        diarization_result_temp = [i.split(' ') for i in str(diarization_result).split('\n') if i.strip() != '']
        locuteurs = list(set([i[6] for i in diarization_result_temp]))

        if output_format == 'eaf':
            eaf = pympi.Elan.Eaf()
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


# Mise à jour de build_dataset_from_eaf pour prendre en entrée un dossier et produire un fichier TSV par fichier EAF

# def build_dataset_from_eaf_dir(eaf_dir, wav_dir, output_base_dir):
#     import os
#     import pympi
#     import pandas as pd
#     from pydub import AudioSegment

#     os.makedirs(os.path.join(output_base_dir, "tsv"), exist_ok=True)
#     annotations_passed = 0

#     for file_name in os.listdir(eaf_dir):
#         if file_name.endswith(".eaf"):
#             base_name = file_name[:-4]  # retirer .eaf
#             eaf_path = os.path.join(eaf_dir, file_name)
#             wav_path = os.path.join(wav_dir, base_name + ".wav")
#             if not os.path.exists(wav_path):
#                 print(f"Fichier WAV manquant pour : {base_name}")
#                 continue

#             eaf = pympi.Elan.Eaf(eaf_path)
#             audio = AudioSegment.from_wav(wav_path)

#             data = []

#             for tier_name in eaf.tiers:
#                 if "_extra" not in tier_name:
#                     for annotation in eaf.get_annotation_data_for_tier(tier_name):
#                         start_time, end_time, value = annotation
#                         duration = (end_time - start_time) / 1000
#                         value = value.strip()
#                         if duration > 30 or (value.startswith("[") and value.endswith("]")):
#                             annotations_passed += 1
#                             continue

#                         extract = audio[start_time:end_time]

#                         # Vérifications robustes
#                         if len(extract) == 0 or extract.frame_count() == 0:
#                             print(f"⚠️ Segment vide ignoré : {base_name} {tier_name} {start_time}-{end_time}")
#                             annotations_passed += 1
#                             continue

#                         try:
#                             extract_dir = os.path.join(output_base_dir, "wav")
#                             os.makedirs(extract_dir, exist_ok=True)
#                             output_wav = os.path.join(extract_dir, f"{base_name}_{tier_name}_{start_time}_{end_time}.wav")
#                             extract.export(output_wav, format="wav")
#                         except Exception as e:
#                             print(f"⚠️ Export échoué pour segment {base_name} {tier_name} {start_time}-{end_time} : {e}")
#                             annotations_passed += 1
#                             continue

#                         timecodes = f"[{start_time}, {end_time}]"
#                         data.append([output_wav, value, timecodes, tier_name])

#             df = pd.DataFrame(data, columns=["audio", "text", "timecodes", "speaker"])
#             output_tsv_path = os.path.join(output_base_dir, "tsv", f"{base_name}.tsv")
#             df.to_csv(output_tsv_path, sep='\t', index=False)
#             print(f"TSV généré : {output_tsv_path}")

#     print(f"Annotations ignorées : {annotations_passed}")
#     return os.path.join(output_base_dir, "tsv")


def build_dataset_from_eaf_dir(eaf_dir, wav_dir, output_base_dir):
    import os
    import pandas as pd
    import pympi
    from pydub import AudioSegment
    from pydub.exceptions import CouldntEncodeError
    import tempfile
    import subprocess

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
def load_whisper_model(model_id):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torch

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model = model.to("cuda")
    print(f"Modèle Whisper chargé : {model_id}")
    return processor, model


# Chargement des datasets audio à partir d'un dossier TSV
def load_dataset_from_tsv_dir(tsv_dir):
    from datasets import load_dataset, Audio, DatasetDict
    import os

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
    import os
    import pandas as pd
    from pydub import AudioSegment
    import torch
    import numpy as np

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
    import os
    import pympi
    import pandas as pd

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
                            # print(f"Annotation mise à jour : {speaker}, {start_time}-{end_time}, {text}")

            output_eaf_path = os.path.join(output_dir, f"{base_name}.eaf")
            eaf.to_file(output_eaf_path)
            print(f"Fichier EAF mis à jour : {output_eaf_path}")

# Fonction pour appel depuis une interface Gradio
def run_pipeline_from_interface(wav_dir, eaf_dir, output_dir, diar_model_id, asr_model_id, hf_token, output_format, language):
    torch.cuda.empty_cache()
    gc.collect()

    if output_format not in ["tsv", "eaf"]:
        return f"Format invalide : {output_format}"

    if diar_model_id and hf_token:
        pipeline = load_diarization_pipeline(diar_model_id, hf_token)
        run_diarization_on_dir(wav_dir, output_dir, pipeline, output_format)
    else:
        return "Veuillez fournir un modèle de segmentation et un token HF."

    if output_format == "eaf":
        if not eaf_dir:
            return "Le dossier EAF est requis pour une sortie au format EAF."
        tsv_dir = build_dataset_from_eaf_dir(eaf_dir, wav_dir, output_dir)
    else:
        tsv_dir = os.path.join(output_dir, "tsv")

    processor, model = load_whisper_model(asr_model_id)
    dataset_dict = load_dataset_from_tsv_dir(tsv_dir)
    transcribed_dir = os.path.join(output_dir, "tsv_transcribed")
    transcribe_dataset_dir(dataset_dict, processor, model, transcribed_dir, language=language)

    if output_format == "eaf" and eaf_dir:
        eaf_updated_dir = os.path.join(output_dir, "eaf_updated")
        update_eaf_with_tsv_dir(eaf_dir, transcribed_dir, eaf_updated_dir)

    return f"✅ Pipeline terminé. Résultats disponibles dans : {output_dir}"


# CLI principale
if __name__ == "__main__":
    import argparse
    import os
    import torch
    import gc

    torch.cuda.empty_cache()
    gc.collect()

    parser = argparse.ArgumentParser(description="Pipeline de diarisation et transcription.")

    parser.add_argument('--mode', type=str, required=True,
                        choices=['diarize', 'transcribe', 'pipeline'],
                        help="Mode d'exécution : diarize, transcribe, pipeline")

    parser.add_argument('--wav-dir', type=str, required=True, help="Dossier contenant les fichiers WAV")
    parser.add_argument('--eaf-dir', type=str, default=None, help="Dossier contenant les fichiers EAF")
    parser.add_argument('--output-dir', type=str, required=True, help="Dossier où sauvegarder les résultats")
    parser.add_argument('--segmentation-model-id', type=str, required=False, help="ID du modèle de segmentation")
    parser.add_argument('--asr-model-id', type=str, required=False, default="openai/whisper-large-v3-turbo", help="ID du modèle ASR")
    parser.add_argument('--hf-token', type=str, required=False, help="Hugging Face token")
    parser.add_argument('--out-format', type=str, choices=['eaf', 'tsv'], default='tsv', help="Format de sortie")
    parser.add_argument('--language', type=str, required=False, help="Code langue pour forcer Whisper (ex: 'fr', 'ht', 'en')")

    args = parser.parse_args()

    if args.mode in ['diarize', 'pipeline']:
        pipeline = load_diarization_pipeline(args.segmentation_model_id, args.hf_token)
        run_diarization_on_dir(args.wav_dir, args.output_dir, pipeline, args.out_format)

    if args.mode == 'transcribe':
        if not args.eaf_dir:
            raise ValueError("--eaf-dir est requis pour la transcription.")
        tsv_dir = build_dataset_from_eaf_dir(args.eaf_dir, args.wav_dir, args.output_dir)
    elif args.mode == 'pipeline':
        tsv_dir = os.path.join(args.output_dir, "tsv")

    if args.mode in ['transcribe', 'pipeline']:
        processor, model = load_whisper_model(args.asr_model_id)
        dataset_dict = load_dataset_from_tsv_dir(tsv_dir)
        transcribed_tsv_dir = os.path.join(args.output_dir, "tsv_transcribed")

        # ✅ Passage du paramètre de langue à la fonction
        transcribe_dataset_dir(dataset_dict, processor, model, transcribed_tsv_dir, language=args.language)

        if args.out_format == "eaf" and args.eaf_dir:
            eaf_updated_dir = os.path.join(args.output_dir, "eaf_updated")
            update_eaf_with_tsv_dir(args.eaf_dir, transcribed_tsv_dir, eaf_updated_dir)

