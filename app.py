import os
import torch
import soundfile as sf
import gradio as gr

from src.model import Seq2SeqTTS
from src.config import Config
from src.preprocessing import get_vocab_size, text_to_sequence, spectrogram_to_audio
from src.enhancement import enhance_audio

# 1. Load the model globally when the app starts
device = torch.device('cpu')
vocab_size = get_vocab_size()
model = Seq2SeqTTS(vocab_size, Config.embed_dim, Config.hidden_dim, Config.spec_bins).to(device)

model_path = 'outputs/checkpoints/best_model.pt' if os.path.exists('outputs/checkpoints/best_model.pt') else 'best_model.pt'

if os.path.exists(model_path):
    print(f"✅ Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
else:
    print("⚠️ WARNING: best_model.pt not found! The app will output untrained noise.")

# 2. Define the synthesis function
def tts_generate(text):
    if not text.strip():
        return None, None
        
    print(f"Synthesizing: '{text}'")
    with torch.no_grad():
        seq = torch.LongTensor(text_to_sequence(text)).unsqueeze(0).to(device)
        seq_lengths = torch.LongTensor([seq.size(1)]).to(device)
        
        # Target length defines the audio duration (~4 seconds here)
        dummy_target = torch.zeros(1, 300, Config.spec_bins).to(device)
        predicted_spec = model(seq, seq_lengths, dummy_target, teacher_forcing_ratio=0.0)
        
    spec_np = predicted_spec.squeeze(0).cpu().numpy()
    
    # Generate Raw Audio
    raw_audio = spectrogram_to_audio(spec_np)
    os.makedirs('outputs/demo', exist_ok=True)
    raw_path = 'outputs/demo/app_raw.wav'
    sf.write(raw_path, raw_audio, Config.sample_rate)
    
    # Generate Enhanced Audio
    enhanced_audio = enhance_audio(raw_audio, Config.sample_rate, method='all')
    enhanced_path = 'outputs/demo/app_enhanced.wav'
    sf.write(enhanced_path, enhanced_audio, Config.sample_rate)
    
    return raw_path, enhanced_path

# 3. Build the beautiful Web UI
custom_css = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
"""

app = gr.Interface(
    fn=tts_generate,
    inputs=gr.Textbox(lines=3, placeholder="Enter text to synthesize...", label="Input Text"),
    outputs=[
        gr.Audio(label="Raw Model Output (Griffin-Lim Artifacts)"),
        gr.Audio(label="Enhanced Output (Wiener Filter & Spectral Gating)")
    ],
    title="🎙️ End-to-End Text-to-Speech Synthesizer",
    description="Type a sentence below to generate speech using our custom Seq2Seq Neural Network. Compare the raw Griffin-Lim output to the digitally enhanced version!",
    examples=[
        "Welcome to our machine learning presentation.",
        "We built an end to end text to speech model.",
        "The neural network consists of over ten million parameters."
    ],
    allow_flagging="never",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print("\n🚀 Starting Web App... Open the Local URL in your browser!")
    app.launch(share=False)
