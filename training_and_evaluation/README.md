\# Training and Evaluation



This folder contains the training, evaluation, inference, reconstruction,

and visualization work carried out for the TTS mini-project.



\## 1. Training Setup



The original LJSpeech dataset contains 13,100 audio samples. Due to

limited local CPU resources, the complete dataset could not be used for

model training.



For the experiments documented here:



\- Dataset used for training: 100 samples

\- Training samples: 80

\- Validation samples: 20

\- Batch size: 2

\- Learning rate: 0.0001

\- Model: Sequence-to-sequence TTS model

\- Embedding dimension: 256

\- Hidden dimension: 512

\- Spectrogram bins: 1025



\## 2. Baseline Training



The baseline model was trained for 3 epochs.



Results:



| Epoch | Training Loss | Validation Loss |

|------:|--------------:|----------------:|

| 1 | 0.079800 | 0.095617 |

| 2 | 0.051656 | 0.106648 |

| 3 | 0.047776 | 0.098452 |



The best baseline checkpoint was saved as:



`checkpoints/seq2seq\_best.pt`



\## 3. Training Improvement: Data Augmentation



An additional training improvement was implemented using spectrogram

augmentation.



The augmentation parameters were:



\- Noise standard deviation: 0.015

\- Time masking ratio: 0.08

\- Frequency masking ratio: 0.04

\- Gain range: 0.9–1.1



Augmentation was applied only to the training data. Validation data

remained unchanged.



\### Augmented Training Results



| Epoch | Training Loss | Validation Loss |

|------:|--------------:|----------------:|

| 1 | 0.076454 | 0.102839 |

| 2 | 0.052748 | 0.099040 |

| 3 | 0.047003 | 0.098764 |



The best augmented checkpoint was saved as:



`checkpoints/seq2seq\_augmented\_best.pt`



\## 4. Reconstruction



Speech reconstruction from an existing spectrogram was successfully

performed.



For example:



\- Input spectrogram: `(699, 1025)`

\- Sample rate: 22050 Hz

\- Reconstructed audio duration: approximately 8.10 seconds



The reconstructed speech was clearly recognizable.



This demonstrates that the spectrogram-to-audio reconstruction portion

of the pipeline was functioning successfully.



\## 5. Text-to-Speech Inference



Text-to-spectrogram inference was also executed using the trained

sequence-to-sequence model.



Example input:



"She had your dark suit in greasy wash water all year."



The model generated a spectrogram with shape:



`(300, 1025)`



The generated spectrogram was saved as:



`results/generated\_spectrogram.npy`



The corresponding WAV output was also generated.



However, with the limited 100-sample CPU training setup, the generated

TTS audio contained noise/buzzing and did not produce clearly

intelligible speech.



Therefore, the current results demonstrate successful execution of

the text-to-spectrogram pipeline, but the generated audio quality is

not yet satisfactory.



\## 6. Evaluation and Visualization



The following scripts are included:



\- `train.py` – model training

\- `inference.py` – text-to-spectrogram inference

\- `evaluate.py` – model evaluation

\- `test\_reconstruction.py` – reconstruction testing

\- `test\_real\_reconstruction.py` – reconstruction from a real spectrogram

\- `plot\_training.py` – training/validation loss visualization

\- `plot\_spectrogram.py` – spectrogram visualization

\- `save\_generated\_spectrogram.py` – saves/visualizes generated spectrograms



\## 7. Files



\### checkpoints/



Contains the trained baseline and augmented model checkpoints.



\### results/



Contains training histories and the generated spectrogram.



\### scripts/



Contains the training, evaluation, inference, reconstruction and

visualization scripts.



\## 8. Limitation



The complete 13,084-sample training subset available after preprocessing

was not trained locally because CPU training was too time-consuming.



The experiments therefore demonstrate the complete pipeline and the

effect of the implemented augmentation technique on a small training

subset, rather than claiming full-dataset TTS performance.

