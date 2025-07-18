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

import gradio as gr
import os
import torch
import gc

from DIAxASR import (
    run_pipeline_from_interface,
)

def interface_pipeline(
    mode,
    wav_dir,
    input_dir,
    output_dir,
    diar_model_id,
    asr_model_id,
    hf_token,
    output_format,
    language
):
    torch.cuda.empty_cache()
    gc.collect()
    try:
        # Le paramètre 'mode' gère toute la logique : "diarize", "transcribe", "pipeline"
        # L'entrée input_dir peut contenir soit des .eaf, soit des .TextGrid, conversion auto dans le pipeline
        # On choisit CUDA si possible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = run_pipeline_from_interface(
            wav_dir=wav_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            diar_model_id=diar_model_id,
            asr_model_id=asr_model_id,
            hf_token=hf_token,
            output_format=output_format,
            language=language,
            device=device,
            mode=mode,
            keep_temp_audio=False
        )
        return result
    except Exception as e:
        return f"❌ Erreur lors du traitement : {str(e)}"

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            mode = gr.Radio(
                choices=["diarize", "transcribe", "pipeline"],
                label="Mode d'exécution",
                value="pipeline"
            )
            wav_dir = gr.Textbox(label="Dossier WAV")
            input_dir = gr.Textbox(label="Dossier EAF/TextGrid (segmentation manuelle ou issue de la diarisation)")
            output_dir = gr.Textbox(label="Dossier de sortie")
            diar_model_id = gr.Textbox(label="Modèle Pyannote (laisser vide pour défaut)")
            asr_model_id = gr.Textbox(label="Modèle Whisper ASR", value="openai/whisper-large-v3-turbo")
            hf_token = gr.Textbox(label="Hugging Face token", type="password")
            output_format = gr.Radio(
                choices=["tsv", "eaf", "textgrid"],
                label="Format de sortie final",
                value="tsv"
            )
            language = gr.Textbox(label="Langue (ex: fr, en, ht)", value="fr")
            submit = gr.Button("Lancer le traitement")

        with gr.Column():
            logo_path = r"DIAxASR/assets/logo_DIAxASR2.png"
            if os.path.exists(logo_path):
                gr.Image(value=logo_path, show_label=False, height=300)
            output_box = gr.Textbox(label="Journal de traitement", lines=20)

    # Mise à jour visibilité champs selon mode sélectionné
    def update_visibility(selected_mode):
        # Pipeline = tout (sauf input_dir qui dépend du format)
        # Diarize = wav_dir, output_dir, diar_model_id, hf_token, output_format
        # Transcribe = wav_dir, input_dir, output_dir, asr_model_id, output_format, language
        return {
            input_dir: gr.update(visible=selected_mode in ["transcribe", "pipeline"]),
            diar_model_id: gr.update(visible=selected_mode in ["diarize", "pipeline"]),
            asr_model_id: gr.update(visible=selected_mode in ["transcribe", "pipeline"]),
            hf_token: gr.update(visible=selected_mode in ["diarize", "pipeline"]),
            output_format: gr.update(visible=True),
            output_dir: gr.update(visible=True),
            language: gr.update(visible=selected_mode in ["transcribe", "pipeline"]),
        }

    mode.change(
        fn=update_visibility,
        inputs=mode,
        outputs=[input_dir, diar_model_id, asr_model_id, hf_token, output_format, output_dir, language]
    )

    submit.click(
        fn=interface_pipeline,
        inputs=[mode, wav_dir, input_dir, output_dir, diar_model_id, asr_model_id, hf_token, output_format, language],
        outputs=output_box
    )

iface.launch()

