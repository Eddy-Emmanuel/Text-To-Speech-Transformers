"""
Fine-tune SpeechT5 for Dutch text-to-speech on the VoxPopuli dataset.

This is the script version of `text-to-speech.ipynb` — the same pipeline, organized
into functions so it can be run end-to-end from the command line:

    python train.py
    python train.py --max-steps 100 --output-dir speecht5_smoke_test   # quick test run
    python train.py --skip-training --checkpoint ./speecht5_finetuned_voxpopuli_nl  # inference only

See README.md for setup instructions.
"""

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import datasets
from huggingface_hub import login
from speechbrain.pretrained import EncoderClassifier
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    pipeline,
)


@dataclass
class Config:
    sampling_rate: int
    model_checkpoint: str
    speech_model: str


DEFAULT_CONFIG = Config(
    sampling_rate=16_000,
    model_checkpoint="microsoft/speecht5_tts",
    speech_model="speechbrain/spkrec-xvect-voxceleb",
)

CHAR_REPLACEMENTS = [
    ("à", "a"),
    ("ç", "c"),
    ("è", "e"),
    ("ë", "e"),
    ("í", "i"),
    ("ï", "i"),
    ("ö", "o"),
    ("ü", "u"),
]

SAMPLE_TEXT = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="speecht5_finetuned_voxpopuli_nl",
                         help="Directory to save checkpoints and the final model.")
    parser.add_argument("--max-steps", type=int, default=4000, help="Max training steps.")
    parser.add_argument("--num-train-epochs", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--push-to-hub", action="store_true", help="Push the fine-tuned model to the Hugging Face Hub.")
    parser.add_argument("--skip-training", action="store_true",
                         help="Skip training and only run the inference demo (requires --checkpoint).")
    parser.add_argument("--checkpoint", default=None,
                         help="Path or Hub id of a fine-tuned model to use for inference "
                              "when --skip-training is set. Defaults to --output-dir.")
    parser.add_argument("--sample-text", default=SAMPLE_TEXT, help="Text to synthesize in the inference demo.")
    return parser.parse_args()


def hf_login_if_available():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("HF_TOKEN not set — continuing without login (fine for public models/datasets).")


def load_and_clean_dataset(config: Config):
    """Load Dutch VoxPopuli, resample audio, and normalize characters missing from the tokenizer vocab."""
    data = datasets.load_dataset("qmeeus/voxpopuli", "nl", split="train")
    data_casted = data.cast_column("audio", datasets.Audio(sampling_rate=config.sampling_rate))

    def cleanup_text(inputs):
        for src, dst in CHAR_REPLACEMENTS:
            inputs["text"] = inputs["text"].replace(src, dst)
        return inputs

    dataset = data_casted.map(cleanup_text)
    return data_casted, dataset


def filter_speakers(data_casted, dataset):
    """Keep speakers with a moderate number of utterances (100-400)."""
    speaker_counts = defaultdict(int)
    for speaker_id in dataset["audio_id"]:
        speaker_counts[speaker_id] += 1

    def select_speaker(speaker_id):
        return 100 <= speaker_counts[speaker_id] <= 400

    return data_casted.filter(select_speaker, input_columns=["audio_id"])


def build_speaker_embedder(config: Config, device: str):
    """Load the pretrained SpeechBrain x-vector model and return an embedding function."""
    speaker_model = EncoderClassifier.from_hparams(
        source=config.speech_model,
        run_opts={"device": device},
        savedir=os.path.join("/tmp", config.speech_model),
    )

    def create_speaker_embedding(waveform):
        with torch.no_grad():
            speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
        return speaker_embeddings

    return create_speaker_embedding


def build_prepared_dataset(dataset, processor, create_speaker_embedding):
    """Tokenize text, extract mel-spectrogram targets, and attach speaker embeddings."""

    def prepare_dataset(example):
        audio = example["audio"]
        example = processor(
            text=example["text"],
            audio_target=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
        )
        # strip off the batch dimension
        example["labels"] = example["labels"][0]
        # use SpeechBrain to obtain x-vector
        example["speaker_embeddings"] = create_speaker_embedding(audio["array"])
        return example

    prepared = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

    def is_not_too_long(input_ids):
        return len(input_ids) < 200

    prepared = prepared.filter(is_not_too_long, input_columns=["input_ids"])
    return prepared


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any
    model: Any

    def __call__(self, features):
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = self.processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if self.model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [length - length % self.model.config.reduction_factor for length in target_lengths]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch


def train(args, config: Config, device: str):
    hf_login_if_available()

    print("Loading dataset...")
    data_casted, dataset = load_and_clean_dataset(config)

    print("Filtering speakers...")
    data_casted_filt = filter_speakers(data_casted, dataset)  # noqa: F841 (mirrors original notebook)

    processor = SpeechT5Processor.from_pretrained(config.model_checkpoint)

    print("Building speaker embeddings and training examples (this can take a while)...")
    embed_fn = build_speaker_embedder(config, device)
    dataset = build_prepared_dataset(dataset, processor, embed_fn)
    dataset = dataset.train_test_split(test_size=0.1)
    print(f"Train examples: {len(dataset['train'])}, eval examples: {len(dataset['test'])}")

    print("Loading model...")
    model = SpeechT5ForTextToSpeech.from_pretrained(config.model_checkpoint)
    # disable cache during training since it's incompatible with gradient checkpointing
    model.config.use_cache = False
    # re-enable cache for generation
    model.generate = partial(model.generate, use_cache=True)

    data_collator = TTSDataCollatorWithPadding(processor=processor, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
        push_to_hub=args.push_to_hub,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=processor,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    return trainer, processor, dataset


def run_inference(model, processor, dataset, device: str, sample_text: str, output_path: str = "sample.wav"):
    """Synthesize `sample_text` using a speaker embedding from the held-out test set."""
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    tts = pipeline(
        "text-to-speech",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        vocoder=vocoder,
    )

    example = dataset["test"][min(304, len(dataset["test"]) - 1)]
    speaker_embeddings = torch.tensor(example["speaker_embeddings"], device=device).unsqueeze(0)

    output = tts(sample_text, forward_params={"speaker_embeddings": speaker_embeddings})

    import soundfile as sf
    sf.write(output_path, output["audio"], samplerate=output["sampling_rate"])
    print(f"Wrote synthesized audio to {output_path}")
    return output


def main():
    args = parse_args()
    config = DEFAULT_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.skip_training:
        checkpoint = args.checkpoint or args.output_dir
        hf_login_if_available()
        print(f"Loading fine-tuned model from {checkpoint} for inference...")
        processor = SpeechT5Processor.from_pretrained(checkpoint)
        model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint).to(device)

        # need a dataset split to pull a sample speaker embedding from
        _, dataset = load_and_clean_dataset(config)
        embed_fn = build_speaker_embedder(config, device)
        dataset = build_prepared_dataset(dataset, processor, embed_fn)
        dataset = dataset.train_test_split(test_size=0.1)

        run_inference(model, processor, dataset, device, args.sample_text)
    else:
        trainer, processor, dataset = train(args, config, device)
        run_inference(trainer.model, processor, dataset, device, args.sample_text)


if __name__ == "__main__":
    main()
