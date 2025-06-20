# DIAxASR – Pipeline de Diarisation et Transcription Automatique

## Présentation

`DIAxASR.py` est un pipeline Python complet permettant :

- la **diarisation automatique** de fichiers audio à l’aide d’un modèle Pyannote Audio ;
- la **segmentation des enregistrements** avec export des segments audio et des métadonnées (`.tsv`, `.eaf`) ;
- la **transcription automatique** avec un modèle Whisper fine-tuné ;
- la **mise à jour des fichiers ELAN (`.eaf`)** avec les transcriptions.

Ce pipeline peut être exécuté :
- en **ligne de commande** via le script `DIAxASR.py` ;
- ou **pas-à-pas** via le notebook interactif `run_DIAxASR.ipynb`.

---

## Arborescence recommandée

```
projet/
├── DIAxASR.py
├── run_DIAxASR.ipynb
├── requirements.txt
├── audio/             # fichiers WAV à traiter
├── eaf/               # (facultatif) fichiers .eaf existants
└── outputs/           # dossiers de sortie générés automatiquement
```

---

## Installation

### Étape 1 – Création d’un environnement Python

Créez un environnement `conda` ou `venv` (Python 3.10+ recommandé) :

```bash
conda create -n diaxasr python=3.11
conda activate diaxasr
```

### Étape 2 – Installation des dépendances

Deux options sont possibles :

#### Option 1 – Installation via le notebook (recommandée pour explorations)

Le notebook `run_DIAxASR.ipynb` contient des **cellules d'installation des dépendances** :

```python
!git clone https://github.com/huggingface/diarizers.git
!pip install -e /home/ziane212/projects/MMS_ASR_finetuning/diarizers/.
```
et 
```python
!pip install -r chemin/du/fichier/requirements.txt
```

Assurez-vous d'avoir modifié le chemin du `requirements.txt`.

#### Option 2 – Installation manuelle

En ligne de commande :

```bash
pip install -r requirements.txt
```

**Le modèle Pyannote nécessite un token d'authentification Hugging Face.**

---

## Exécution

### En notebook (`run_DIAxASR.ipynb`)

Ce notebook exécute étape par étape :

1. La **diarisation** avec export TSV ou EAF ;
2. La **création du jeu de données** à partir de fichiers `.eaf` ou `.tsv` ;
3. La **transcription automatique** avec Whisper ;
4. La **mise à jour facultative** des fichiers `.eaf` avec les transcriptions.

**Idéal pour tester sur un sous-ensemble et prendre en main le script.**

---

### En ligne de commande

Vous pouvez aussi exécuter le script complet via :

```bash
python DIAxASR.py \
  --mode pipeline \
  --wav-dir ./audio \
  --eaf-dir ./eaf \
  --output-dir ./outputs \
  --segmentation-model-id your_model_id \
  --hf-token your_hf_token \
  --asr-model-id Rziane/whisper-large-v3-turbo-CAENNAIS \
  --out-format eaf
```

Autres modes possibles :
- `--mode diarize` : diarisation uniquement
- `--mode transcribe` : transcription uniquement (à partir de fichiers `.eaf` qui contiennent des tours de parole segmentés par locuteurs)

---

## Résultats produits

Selon les paramètres et les formats choisis :

- `outputs/tsv/` : métadonnées après diarisation ;
- `outputs/tsv_transcribed/` : transcriptions par segments ;
- `outputs/eaf/` : fichiers ELAN après diarisation (si `--out-format eaf`) ;
- `outputs/eaf_updated/` : fichiers ELAN mis à jour avec les transcriptions.

---

## Remarques

- La diarisation est actuellement configurée pour **2 locuteurs** (`num_speakers=2`), mais cela peut être adapté.
- Les segments de plus de 30s sont ignorés pour éviter les erreurs ASR.
- Le script crée automatiquement tous les sous-dossiers nécessaires.

---

## Auteurs

Développé par **Rayan Ziane** dans le cadre du projet [CAENNAIS](https://crisco.unicaen.fr/caennais-corpus-audio-detudiants-natifs-et-non-natifs-en-interactions/): 

Modèle ASR pour la transcription semi-orthographique du français parlé : [`Rziane/whisper-large-v3-turbo-CAENNAIS_GB`](https://huggingface.co/Rziane/whisper-large-v3-turbo-CAENNAIS_GB)


---

## Références

- **Whisper (ASR)** : OpenAI. [https://github.com/openai/whisper](https://github.com/openai/whisper)  
  Modèle utilisé par défaut : [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)

- **pyannote-audio (Diarisation)** : Hervé Bredin et al. [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)

- **ELAN** : Max Planck Institute for Psycholinguistics. [https://archive.mpi.nl/tla/elan](https://archive.mpi.nl/tla/elan)

