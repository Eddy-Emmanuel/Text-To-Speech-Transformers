# Text-To-Speech-Transformers

### Dutch Text-to-Speech — Fine-Tuning SpeechT5

Fine-tunes Microsoft's [SpeechT5](https://huggingface.co/microsoft/speecht5_tts) text-to-speech model on the
Dutch (`nl`) split of the [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) dataset, producing a
multi-speaker Dutch TTS model conditioned on x-vector speaker embeddings.

Given a sentence of Dutch text and a speaker embedding, the fine-tuned model + a HiFi-GAN vocoder synthesize
the corresponding speech.

## How it works

1. Load the Dutch VoxPopuli dataset and resample audio to 16 kHz.
2. Clean the transcripts (normalize accented characters not in SpeechT5's tokenizer vocabulary).
3. Keep speakers with a moderate number of utterances (100–400) for a more balanced dataset.
4. Extract a speaker embedding per utterance with a pretrained [SpeechBrain x-vector model](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb).
5. Tokenize text and extract mel-spectrogram targets with the SpeechT5 processor.
6. Fine-tune `SpeechT5ForTextToSpeech` with Hugging Face's `Seq2SeqTrainer`.
7. Run inference with the fine-tuned model and the pretrained HiFi-GAN vocoder.

## Requirements

- Python 3.10+
- A CUDA GPU (training uses `fp16=True`; a single 16 GB GPU such as a T4 is enough, but training will take
  several hours at the default settings)
- A Hugging Face account/token (optional, but recommended to avoid rate limits — see below)

## Setup

```bash
git clone https://github.com/Eddy-Emmanuel/Text-To-Speech-Transformers.git
cd Text-To-Speech-Transformers
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Hugging Face authentication

The model and dataset used here are public, but logging in avoids rate limiting and is required if you want
to push your fine-tuned model to the Hub. Set your token as an environment variable before launching Jupyter:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

(On Windows PowerShell: `$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"`)

Alternatively, run `huggingface-cli login` once and skip this step entirely — the notebook falls back
gracefully if `HF_TOKEN` isn't set.

## Usage

Two equivalent ways to run the pipeline: the notebook (for exploring step by step) or `train.py` (for a
straight-through run, e.g. on a remote GPU box).

### Notebook

```bash
jupyter lab text-to-speech.ipynb
```

Run the cells top to bottom. The notebook is organized into numbered sections:

| Section | What it does |
|---|---|
| 1. Environment setup | GPU check, install `speechbrain`, HF login |
| 2. Configuration | Model checkpoint, sampling rate, speaker-embedding model |
| 3. Load the dataset | Pull Dutch VoxPopuli from the Hub |
| 4. Text cleanup | Normalize characters missing from the tokenizer vocab |
| 5. Filter speakers | Keep speakers with 100–400 utterances |
| 6. Speaker embeddings | Compute x-vectors with SpeechBrain |
| 7. Build training examples | Tokenize + extract spectrogram targets |
| 8. Data collator | Batch padding, loss masking, reduction-factor rounding |
| 9. Load the model | `SpeechT5ForTextToSpeech` from the pretrained checkpoint |
| 10. Training arguments | `Seq2SeqTrainingArguments` |
| 11. Train | `trainer.train()` — the long-running step |
| 12. Inference | Synthesize a sample sentence and play the audio |

### Script

`train.py` mirrors the notebook as a runnable CLI script:

```bash
# Full fine-tuning run with the default settings
python train.py

# Quick smoke test — lower step count, separate output dir
python train.py --max-steps 100 --save-steps 25 --eval-steps 25 --output-dir speecht5_smoke_test

# Inference only, using an already fine-tuned checkpoint
python train.py --skip-training --checkpoint ./speecht5_finetuned_voxpopuli_nl --sample-text "goedemorgen!"
```

Run `python train.py --help` for the full list of options (batch size, learning rate, warmup steps, etc.).
Training saves the final model and processor to `--output-dir`, and both modes write a synthesized sample to
`sample.wav` in the working directory.

### Quick test run

The default `max_steps=4000` is meant for a real fine-tuning run and can take several hours. To sanity-check
the pipeline end-to-end first, lower `max_steps` (e.g. to `100`) and `save_steps` / `eval_steps` accordingly —
either in the training-arguments cell of the notebook, or via `--max-steps` / `--save-steps` / `--eval-steps`
on the command line.

## Notes

- The original notebook was developed on Kaggle with `kaggle_secrets` for token storage; this version uses a
  plain `HF_TOKEN` environment variable so it runs anywhere.
- Training checkpoints are written to `speecht5_finetuned_voxpopuli_nl/` (excluded from version control — see
  `.gitignore`).
- `push_to_hub` is disabled by default in the training arguments; set it to `True` (and adjust `output_dir` to
  a Hub repo name) if you want to publish the fine-tuned model.

## Acknowledgements

- [SpeechT5](https://arxiv.org/abs/2110.07205) (Microsoft)
- [VoxPopuli](https://arxiv.org/abs/2101.00390) dataset
- [SpeechBrain](https://speechbrain.github.io/) x-vector speaker recognition model
- Based on the Hugging Face [TTS fine-tuning guide](https://huggingface.co/learn/audio-course/chapter6/fine-tuning)

## License

Add a license of your choice (e.g. MIT) — see `LICENSE`.
