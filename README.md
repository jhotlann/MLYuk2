# MLYuk2

Repository untuk Tugas Besar IF3270: implementasi CNN, Simple RNN, dan LSTM _from scratch_ (NumPy) serta pipeline image captioning berbasis encoder-decoder (Keras untuk training). Dataset utama adalah Intel Image Classification (folder dataset/) dan Flickr8k (folder data/).

## Struktur Singkat
- src/cnn: layer CNN from scratch (conv, pooling, flatten, activation, dll).
- src/rnn: layer RNN/LSTM from scratch dan model captioning.
- src/captioning/preprocessing.py: preprocessing caption Flickr8k.
- notebooks/: training, evaluasi, dan pipeline end-to-end.
- models/rnn/: model Keras hasil training untuk decoder RNN/LSTM.

## Setup
1. Buat dan aktifkan virtual environment.
	```bash
	python -m venv venv
	source venv/bin/activate
	```
2. Install dependencies.
	```bash
	pip install -r requirements.txt
	```

## Cara Menjalankan
Melalui Notebook:
- rnn_training.ipynb: training 12 variasi decoder RNN/LSTM pada Flickr8k.
- rnn_evaluation.ipynb: evaluasi BLEU-4 + METEOR, perbandingan Keras vs Scratch.
- rnn_pipeline.ipynb: pipeline end-to-end (image -> CNN feature -> scratch decoder).

## Pembagian Tugas
| Nama | NIM | Tugas |
| --- | --- | --- |
| Anggota 1 | NIM1 | Placeholder tugas |
| Anggota 2 | NIM2 | Placeholder tugas |
| Anggota 3 | NIM3 | Placeholder tugas |
