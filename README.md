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
- cnn_training.ipynb: training 16 variase encoder CNN
- cnn_evaluation.ipynb: evaluasi dan analisis perbandingan Keras vs Scratch, shared vs non-shared
- cnn_hyperparameter.ipynb: analisis pengaruh hyperparameter
- rnn_training.ipynb: training 12 variasi decoder RNN/LSTM pada Flickr8k.
- rnn_evaluation.ipynb: evaluasi BLEU-4 + METEOR, perbandingan Keras vs Scratch.
- rnn_pipeline.ipynb: pipeline end-to-end (image -> CNN feature -> scratch decoder).

## Pembagian Tugas
| Nama                      | NIM      | Tugas                                                                                 |
| --------------------------| ---------| ------------------------------------------------------------------------------------- |
| Joel Hotlan Haris Siahaan | 13523025 | CNN: utility, forward propagation, pelatihan model, analisis dan evaluasi, laporan    |
| Julius Arthur             | 13523030 | Encoder, feature extraction untuk RNN/LSTM, Encoder Decoder pipeline, laporan         |
| Salman Hanif              | 13523056 | RNN/LSTM: forward, preprocessing, pelatihan decoder, analisis RNN vs LSTM, laporan    |
