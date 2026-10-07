import os
import sys
import torch
import soundfile as sf
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Seq2SeqTTS
from src.config import Config
from src.preprocessing import get_vocab_size, text_to_sequence, spectrogram_to_audio

def main():
    parser = argparse.ArgumentParser(description="Test TTS Model Locally")
    parser.add_argument('--model', type=str, default='best_model.pt', help='Path to the .pt model file')
    parser.add_argument('--text', type=str, default="Hello, this is a local test of my machine learning project.", help='Text to speak')
    parser.add_argument('--output', type=str, default='outputs/samples/local_test.wav', help='Output wav file path')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Could not find model at '{args.model}'")
        print("Please download best_model.pt from Google Drive and place it in the same folder as this script (or provide the path with --model)")
        return

    print("Loading model...")
    device = torch.device('cpu') # Use CPU for local testing
    vocab_size = get_vocab_size()
    
    model = Seq2SeqTTS(vocab_size, Config.embed_dim, Config.hidden_dim, Config.spec_bins).to(device)
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded from Epoch {checkpoint['epoch']}!")

    print(f"Generating audio for text: '{args.text}'")
    with torch.no_grad():
        seq = torch.LongTensor(text_to_sequence(args.text)).unsqueeze(0).to(device)
        seq_lengths = torch.LongTensor([seq.size(1)]).to(device)
        
        # Dummy target for length (generate ~3 seconds of audio)
        dummy_target = torch.zeros(1, 300, Config.spec_bins).to(device)
        
        predicted_spec = model(seq, seq_lengths, dummy_target, teacher_forcing_ratio=0.0)
        
    spec_np = predicted_spec.squeeze(0).cpu().numpy()
    audio = spectrogram_to_audio(spec_np)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sf.write(args.output, audio, Config.sample_rate)
    print(f"✅ Audio saved to {args.output}")
    print("\nTo make it sound better, run the enhancement script:")
    print(f"python scripts/enhance_demo.py --input {args.output}")

if __name__ == "__main__":
    main()
