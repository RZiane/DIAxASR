import gradio as gr
from DIAxASR import (
    run_diarization_on_dir,
    build_dataset_from_eaf_dir,
    load_diarization_pipeline,
    load_whisper_model,
    load_dataset_from_tsv_dir,
    transcribe_dataset_dir,
    update_eaf_with_tsv_dir,
    run_pipeline_from_interface
)
import os
import torch
import gc

def interface_pipeline(mode, wav_dir, eaf_dir, output_dir, diar_model_id, asr_model_id, hf_token, output_format, language):
    torch.cuda.empty_cache()
    gc.collect()

    try:
        if mode == "diarize":
            if not hf_token:
                return "❌ Le token HF est requis pour la diarisation."
            pipeline = load_diarization_pipeline(diar_model_id, hf_token)
            run_diarization_on_dir(wav_dir, output_dir, pipeline, output_format)
            return f"✅ Diarisation terminée. Résultats dans : {output_dir}/{output_format}"

        elif mode == "transcribe":
            if not eaf_dir:
                return "❌ Le dossier EAF est requis pour le mode transcription."
            if not asr_model_id:
                return "❌ Le modèle ASR est requis pour la transcription."
            processor, model = load_whisper_model(asr_model_id)
            tsv_dir = build_dataset_from_eaf_dir(eaf_dir, wav_dir, output_dir)
            dataset_dict = load_dataset_from_tsv_dir(tsv_dir)
            transcribed_dir = os.path.join(output_dir, "tsv_transcribed")
            transcribe_dataset_dir(dataset_dict, processor, model, transcribed_dir, language=language)

            if output_format == "eaf":
                eaf_updated_dir = os.path.join(output_dir, "eaf_updated")
                update_eaf_with_tsv_dir(eaf_dir, transcribed_dir, eaf_updated_dir)

            return f"✅ Transcription terminée. Résultats dans : {transcribed_dir}"

        elif mode == "pipeline":
            if not hf_token:
                return "❌ Le token HF est requis pour le pipeline."
            if not asr_model_id:
                return "❌ Le modèle ASR est requis pour le pipeline."
            pipeline = load_diarization_pipeline(diar_model_id, hf_token)
            run_diarization_on_dir(wav_dir, output_dir, pipeline, output_format)

            if output_format == "eaf":
                if not eaf_dir:
                    return "❌ Le dossier EAF est requis pour une sortie au format EAF."
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

            return f"✅ Pipeline complet terminé. Résultats dans : {output_dir}"

        else:
            return "❌ Mode non reconnu."

    except Exception as e:
        return f"❌ Erreur lors du traitement : {str(e)}"

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            mode = gr.Radio(choices=["diarize", "transcribe", "pipeline"], label="Mode d'exécution", value="pipeline")
            wav_dir = gr.Textbox(label="Dossier WAV")
            eaf_dir = gr.Textbox(label="Dossier EAF (si nécessaire)")
            output_dir = gr.Textbox(label="Dossier de sortie")
            diar_model_id = gr.Textbox(label="Modèle de segmentation Pyannote (laisser vide pour défaut)")
            asr_model_id = gr.Textbox(label="Modèle Whisper ASR", value="openai/whisper-large-v3-turbo")
            hf_token = gr.Textbox(label="Hugging Face token", type="password")
            output_format = gr.Radio(choices=["tsv", "eaf"], label="Format de sortie de la diarisation", value="tsv")
            language = gr.Textbox(label="Langue (ex: fr, en, ht)", value="fr")
            submit = gr.Button("Lancer le traitement")

        with gr.Column():
            logo_path = r"DIAxASR/assets/logo_DIAxASR2.png"
            gr.Image(value=logo_path, show_label=False, height=300)
            output_box = gr.Textbox(label="Journal de traitement", lines=20)

    def update_visibility(selected_mode):
        return {
            diar_model_id: gr.update(visible=selected_mode in ["diarize", "pipeline"]),
            hf_token: gr.update(visible=selected_mode in ["diarize", "pipeline"]),
            eaf_dir: gr.update(visible=selected_mode in ["transcribe", "pipeline"]),
            asr_model_id: gr.update(visible=selected_mode in ["transcribe", "pipeline"]),
            output_format: gr.update(visible=True),
            output_dir: gr.update(visible=True),
            language: gr.update(visible=selected_mode in ["transcribe", "pipeline"])
        }

    mode.change(
        fn=update_visibility,
        inputs=mode,
        outputs=[diar_model_id, hf_token, eaf_dir, asr_model_id, output_format, output_dir, language]
    )

    submit.click(
        fn=interface_pipeline,
        inputs=[mode, wav_dir, eaf_dir, output_dir, diar_model_id, asr_model_id, hf_token, output_format, language],
        outputs=output_box
    )

iface.launch()
