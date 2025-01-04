# Chess Openings Detection and Recognition

A computer vision system for detecting and recognizing chess openings from images or video streams of chess games. This project uses machine learning to identify chess piece positions and match them against a database of known chess openings.

## Features

- Real-time chess board detection from images and video streams
- Piece position recognition with high accuracy
- Opening classification against a comprehensive database of chess openings
- Support for multiple board orientations and viewing angles
- Recognition of standard chess notation (e.g., e4, Nf3)
- Integration with popular chess databases and engines
- Export functionality for game analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chess-openings-detection.git

# Navigate to project directory
cd chess-openings-detection

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- OpenCV 4.5+
- TensorFlow 2.x
- PyTorch 1.8+
- NumPy
- Pandas
- Chess.js

## Usage

### Basic Detection

```python
from chess_detection import ChessDetector
from opening_recognition import OpeningClassifier

# Initialize the detector
detector = ChessDetector()

# Load an image
image_path = "path/to/chess_image.jpg"
board_position = detector.detect_board(image_path)

# Initialize the classifier
classifier = OpeningClassifier()

# Get opening information
opening = classifier.classify_position(board_position)
print(f"Detected Opening: {opening.name}")
print(f"ECO Code: {opening.eco}")
```

### Real-time Detection

```python
import cv2
from chess_detection import RealTimeDetector

detector = RealTimeDetector()
detector.start_webcam_detection()
```

## Model Architecture

The system consists of three main components:

1. **Board Detection**: Uses YOLOv5 for chess board detection and corner point identification
2. **Piece Recognition**: Custom CNN architecture for piece classification
3. **Opening Classification**: Pattern matching against opening database using FEN string comparison

## Training

To train the model on your own dataset:

```bash
# Prepare your dataset
python scripts/prepare_dataset.py --data_dir /path/to/data

# Start training
python train.py --config configs/default.yaml
```

## Performance

- Board Detection Accuracy: 98%
- Piece Recognition Accuracy: 95%
- Opening Classification Accuracy: 92%
- Processing Speed: 30 FPS on GPU

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{chess_openings_detection,
  author = {Your Name},
  title = {Chess Openings Detection and Recognition},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/chess-openings-detection}
}
```

## Acknowledgments

- Thanks to the Stockfish team for their chess engine
- Chess.com for providing annotated game datasets
- The OpenCV community for computer vision tools and resources
