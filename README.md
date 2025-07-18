# DIAxASR – Pipeline de Diarisation et Transcription Multiformat (EAF / TSV / TextGrid)

DIAxASR est un pipeline Python pour la segmentation, la diarisation et la transcription automatique de corpus oraux, avec compatibilité complète entre les formats **ELAN (.eaf)**, **Praat (.TextGrid)** et **TSV**. Il prend en charge la conversion entre ces formats et intègre une interface utilisateur via Gradio (`interface.py`), en plus de la ligne de commande classique.

Développé dans le cadre du projet [CAENNAIS](https://crisco.unicaen.fr/caennais-corpus-audio-detudiants-natifs-et-non-natifs-en-interactions/).

---

## Fonctionnalités principales

* Diarisation automatique à l’aide d’un modèle Pyannote Audio.
* Découpage des fichiers audio et export des segments en `.tsv` ou `.eaf`.
* Transcription automatique des segments audio avec Whisper (modèle fine-tuné compatible).
* Conversion automatique et bidirectionnelle entre les formats ELAN (`.eaf`) et Praat (`.TextGrid`).
* Mise à jour intelligente des fichiers ELAN avec les transcriptions obtenues.
* Filtrage automatique des segments trop longs (>30s) pour fiabiliser l’ASR.
* Interface graphique web avec Gradio (`interface.py`) pour exécuter le pipeline sans coder.
* Gestion automatique des dossiers temporaires et des conversions intermédiaires.
* Support CPU et GPU, détection automatique du device.

---

## Arborescence recommandée

```
projet/
├── DIAxASR.py
├── interface.py
├── run_DIAxASR.ipynb
├── requirements.txt
├── audio/             # fichiers audio à traiter (wav, mp3, etc.)
├── eaf/               # fichiers .eaf existants (facultatif)
├── textgrid/          # fichiers .TextGrid existants (facultatif)
└── outputs/           # dossiers de sortie générés automatiquement
```
---

## Utilsation par notebook

Le notebook `run_DIAxASR.ipynb` permet l'utilisation complète du module de l'installation à l'exécution.
Autrement, la marche à suivre est détaillée ci-dessous:

---

## Installation

### 1. Création de l’environnement Python

```bash
conda create -n diaxasr python=3.11
conda activate diaxasr
```

### 2. Installation des dépendances

```bash
pip install -r requirements.txt
```

Pour utiliser un modèle de segmentation fine-tuné compatible avec [diarizers](https://github.com/huggingface/diarizers) :

```bash
git clone https://github.com/huggingface/diarizers.git
pip install -e diarizers/.
```

Le pipeline Whisper et pyannote nécessite un Hugging Face token (`--hf-token`).

---

## Exécution en ligne de commande

Modes disponibles :

* `--mode diarize` : segmentation et diarisation uniquement, export en `tsv`, `eaf` ou `textgrid`
* `--mode transcribe` : transcription à partir d’un dossier `.eaf` ou `.TextGrid` (conversion automatique)
* `--mode pipeline` : exécution complète de l’audio brut jusqu’à l’export des transcriptions

Exemple d’exécution complète :

```bash
python DIAxASR.py \
  --mode pipeline \
  --wav-dir ./audio \
  --input-dir ./eaf \
  --output-dir ./outputs \
  --segmentation-model-id "votre_modele_pyan" \
  --hf-token "hf_xxxxxxxx" \
  --asr-model-id "openai/whisper-large-v3-turbo" \
  --out-format textgrid
```

Le paramètre `--input-dir` accepte des dossiers de `.eaf` ou de `.textgrid` (conversion automatique).

---

## Utilisation par interface web

L’interface Gradio (`interface.py`) permet d’exécuter tout le pipeline depuis un navigateur (local ou distant).

Lancer simplement :

```bash
python interface.py
```

Fonctionnalités proposées :

* Sélection du mode (`pipeline`, `diarize`, `transcribe`)
* Choix du dossier audio, de l’entrée segmentée (EAF ou TextGrid), du dossier de sortie
* Paramètres de modèles (Pyannote, Whisper, HF token, langue, format de sortie)
* Retour direct du journal de traitement

Tout est accessible par formulaire, aucun chemin n’a besoin d’être codé en dur.

---

## Entrées et sorties prises en charge

* **Entrée** : dossier audio (`wav`, `mp3`, etc.), dossier d’annotations ELAN (`.eaf`) ou Praat (`.TextGrid`)
* **Sortie** : `tsv`, `eaf`, `textgrid` (choix libre selon votre workflow)
* Conversion automatique : vous pouvez fournir uniquement des `.eaf` ou `.TextGrid`, la conversion s’effectue dans le pipeline
* Dossiers produits :

  * `outputs/tsv/` : métadonnées de segmentation/diarisation
  * `outputs/eaf/` : EAF issus de la diarisation
  * `outputs/textgrid_updated/` : TextGrid générés automatiquement
  * `outputs/tsv_transcribed/` : transcriptions segmentées
  * `outputs/eaf_updated/` : EAF mis à jour avec les transcriptions

---

## Points importants

* Les segments de plus de 30 secondes sont ignorés pour la transcription automatique.
* La mise à jour des EAF ne modifie que les intervalles vides ou balisés “!” dans les fichiers TEXTGRID (où les intervalles entre les annotations réelles sont considérés comme des annotations).
* Le script crée tous les sous-dossiers nécessaires automatiquement.
* Vous pouvez utiliser vos propres modèles fine-tunés Pyannote/Whisper.
* Le pipeline détecte automatiquement le matériel pour utiliser votre carte graphique ou le processeur (CPU ou GPU).

---

## Références et auteurs

Développé par Rayan Ziane dans le cadre du projet [CAENNAIS](https://crisco.unicaen.fr/caennais-corpus-audio-detudiants-natifs-et-non-natifs-en-interactions/).

* ASR : [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) ou [Rziane/whisper-large-v3-turbo-CAENNAIS_GB](https://huggingface.co/Rziane/whisper-large-v3-turbo-CAENNAIS_GB)
* Diarisation : [pyannote-audio](https://github.com/pyannote/pyannote-audio)
* ELAN : [MPI ELAN](https://archive.mpi.nl/tla/elan)
* Praat : [Praat](https://www.fon.hum.uva.nl/praat/)
