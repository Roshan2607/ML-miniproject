# End-to-End Text to Speech Synthesis
**Machine Learning Mini-Project Write-Up**

## 1. Problem Statement
The goal of this project is to synthesize natural-sounding human speech directly from raw text. Traditional Text-to-Speech (TTS) systems rely on complex, multi-stage pipelines—requiring separate linguistic feature extractors, acoustic models, and parametric vocoders—which are difficult to build, tune, and maintain. We aim to bypass this complexity by implementing an end-to-end machine learning approach that directly maps text characters and phonemes to audio spectrograms using neural networks. By exploring sequence-to-sequence architectures, we seek to improve the naturalness of the synthesized voice and demonstrate that a unified model can learn the alignment between written language and human speech.

## 2. Dataset Details
We utilized the publicly available **LJSpeech-1.1 dataset**, a standard benchmark in TTS research. 
* **Size:** It contains 13,100 short audio clips (ranging from 1 to 10 seconds each), totaling approximately 24 hours of audio. 
* **Content:** The dataset features a single female speaker reading passages from various non-fiction books.
* **Format:** The data includes raw `.wav` audio files (recorded at a sample rate of 22,050 Hz) and a `metadata.csv` file that provides the exact text transcript for every corresponding audio clip. 

## 3. Our Approach
To solve the problem, we divided the architecture into three phases: Preprocessing, Model Generation, and Audio Reconstruction/Enhancement.

* **Preprocessing:** Raw text was converted into phonetic sequences using the CMU Pronouncing Dictionary (`g2p_en`) to capture true pronunciation rather than just spelling. The raw audio waves were passed through a pre-emphasis filter to amplify high frequencies, and then transformed into 1025-dimensional Mel-spectrograms using the Short-Time Fourier Transform (STFT).
* **Baselines:** We first implemented Support Vector Regression (SVR) and a Simple 4-layer Feedforward Neural Network to map text tokens directly to spectrogram frames. These served as baseline metrics.
* **Main Architecture:** Our primary solution is a **Sequence-to-Sequence (Seq2Seq) neural network with Attention**. 
  * *Encoder:* A Bidirectional LSTM processes the input phoneme sequence into rich contextual embeddings.
  * *Decoder:* An autoregressive LSTM generates the spectrogram one frame at a time.
  * *Attention Mechanism:* We integrated Bahdanau Attention, allowing the decoder to dynamically "focus" on different parts of the input text as it speaks, effectively learning the alignment between specific phonemes and audio frames.
* **Reconstruction & Enhancement:** The model outputs a predicted spectrogram, which is converted back into a listenable waveform using the Griffin-Lim algorithm. To counter the robotic "hiss" typically introduced by Griffin-Lim, we designed a custom audio enhancement module applying Spectral Gating and Wiener Filtering to denoise the final audio.

## 4. Implementation Overview
The entire pipeline was implemented in Python. We utilized **PyTorch** for constructing and training the neural networks and **Librosa** for signal processing. 

To handle variable-length sentences and audio clips, we built custom PyTorch `Dataset` classes with a specific `collate_fn` that applied zero-padding to standardize batch sizes. During training, we utilized **Masked Mean Squared Error (MSE) loss** to ensure the model was only penalized for errors on actual speech frames, ignoring the zero-padded silence. We also employed **Teacher Forcing** (at a 50% ratio) during the early stages of the decoder's training to stabilize learning. 

Because of the massive computational requirement (over 10.7 million parameters processing 13,100 files), training loops were migrated to Google Colab to utilize NVIDIA GPU acceleration, while inference and audio enhancement scripts were configured to run locally.

## 5. Conclusions
Our Seq2Seq model with Bahdanau Attention significantly outperformed the SVR and Simple NN baselines, both in minimizing the Masked MSE reconstruction loss and in generating recognizable speech patterns. 

While the end-to-end Seq2Seq pipeline was successfully built, Google Colab GPU timeouts prevented the model from training beyond Epoch 7. As a result, the Bahdanau Attention mechanism could not fully converge, causing the decoder to generate overly smoothed spectrograms. This resulted in characteristic Griffin-Lim phase artifacts (an 'underwater' effect) during audio reconstruction. 

Despite hardware limitations preventing a perfectly human-like output, we successfully proved that a unified neural network can effectively learn the highly complex, non-linear mapping between written text and human speech. Furthermore, we discovered that robust data preprocessing and custom audio post-processing (Wiener filtering and spectral subtraction) were just as vital to the final audio quality as the core neural architecture. Future work requires dedicated hardware to train for 100+ epochs to achieve crisp phoneme alignment.
