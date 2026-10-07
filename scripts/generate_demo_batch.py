import os
import sys
import torch
import soundfile as sf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Seq2SeqTTS
from src.config import Config
from src.preprocessing import get_vocab_size, text_to_sequence, spectrogram_to_audio
from src.enhancement import enhance_audio

def main():
    # List of custom sentences for your presentation
    # Feel free to edit these strings to whatever you want the model to say!
    demo_sentences = [
        "Hello, welcome to our machine learning mini project.",
        "We built an end to end text to speech synthesis model.",
        "This audio was generated entirely by a neural network.",
        "We used spectral gating and wiener filtering to clean the audio.",
        "Thank you for listening to our presentation."
    ]

    # Look for the model file
    model_paths = ['best_model.pt', 'outputs/checkpoints/best_model.pt']
    model_path = None
    for p in model_paths:
        if os.path.exists(p):
            model_path = p
            break
    
    if not model_path:
        print("❌ Error: Could not find best_model.pt!")
        print("Please make sure it is in your main project folder or the outputs/checkpoints folder.")
        return

    print(f"Loading model from {model_path}...")
    device = torch.device('cpu')
    vocab_size = get_vocab_size()
    
    model = Seq2SeqTTS(vocab_size, Config.embed_dim, Config.hidden_dim, Config.spec_bins).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    output_dir = 'outputs/demo'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🚀 Starting Batch Generation...\n")
    
    for i, text in enumerate(demo_sentences, 1):
        print(f"[{i}/{len(demo_sentences)}] Generating: '{text}'")
        
        with torch.no_grad():
            seq = torch.LongTensor(text_to_sequence(text)).unsqueeze(0).to(device)
            seq_lengths = torch.LongTensor([seq.size(1)]).to(device)
            
            # Dummy target to force the model to generate ~4 seconds of audio
            dummy_target = torch.zeros(1, 300, Config.spec_bins).to(device)
            predicted_spec = model(seq, seq_lengths, dummy_target, teacher_forcing_ratio=0.0)
            
        spec_np = predicted_spec.squeeze(0).cpu().numpy()
        
        # 1. Generate Raw Audio
        raw_audio = spectrogram_to_audio(spec_np)
        raw_path = os.path.join(output_dir, f"sentence_{i}_raw.wav")
        sf.write(raw_path, raw_audio, Config.sample_rate)
        
        # 2. Automatically Apply Enhancement Pipeline
        enhanced_audio = enhance_audio(raw_audio, Config.sample_rate, method='all')
        enhanced_path = os.path.join(output_dir, f"sentence_{i}_enhanced.wav")
        sf.write(enhanced_path, enhanced_audio, Config.sample_rate)
        
        print(f"   -> Saved: {raw_path}")
        print(f"   -> Saved: {enhanced_path}\n")

    print(f"✅ All done! Check the '{output_dir}' folder for your final presentation files.")

if __name__ == "__main__":
    main()
