# HeartsOnFire - Automatic Drum Transcription (ADT)

HeartsOnFire is a project aimed at creating an Automatic Drum Transcription (ADT) system to help beginner drummers create drum scores. This repository contains a series of Jupyter notebooks for generating your own dataset, training the ADT model, and using the trained model to transcribe drum tracks from input .mp3 files into labeled MIDI files.

## Notebooks Included

- `generating our own dataset.ipynb`: This notebook provides instructions and code for generating your own iterative dataset for training the ADT model.
- `model training.ipynb`: This notebook guides you through the process of training the ADT model using the generated dataset.
- `main.ipynb`: This notebook allows you to input an .mp3 file and use the trained ADT model to receive a labeled MIDI file for the source-separated drum track.

## Getting Started

To get started with the HeartsOnFire ADT project, follow these steps:

1. Clone the repository to your local machine:
```bash
git clone https://github.com/skittree/HeartsOnFire.git
```
2. Install [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) and then PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
3. Install the rest of the required dependencies by running the following command:
```bash
pip install -r requirements.txt
```
4. Run `main.ipynb` and follow the instructions to generate MIDIs from mp3 files with drums.

Note: Make sure to download the pre-trained model file `HeartsOnFire-v.1.0.1_nfft1024_88.06.pth` from the models folder in the repository before running the `main.ipynb` notebook.

## Contributions

We welcome contributions to the HeartsOnFire project! If you would like to contribute, please open an issue or submit a pull request on GitHub. We appreciate your feedback and support in making this project better.

## License

This project is released under the [MIT License](https://opensource.org/license/mit/), which means it is open-source and free to use, modify, and distribute for personal and commercial purposes.

## Acknowledgements

We would like to express our gratitude to the following resources, libraries, and tools that have been used in this project:

- [Librosa](https://librosa.org/doc/main/index.html) - for audio processing and feature extraction.
- [PyTorch](https://pytorch.org/) - for deep learning model training.
- [NumPy](https://numpy.org/) - for numerical computing in Python.
- [pretty_midi](https://craffel.github.io/pretty-midi/) - for creating MIDI files in Python.
- [demucs](https://github.com/facebookresearch/demucs) - for source separating drum tracks from input .mp3 files.
- [torch_audiomentations](https://github.com/asteroid-team/torch-audiomentations) - for data augmentation during model training.
- [Pandas](https://pandas.pydata.org/) - for data manipulation and analysis in Python.
- [GitHub](https://github.com/) - for providing a collaborative platform for open-source development.
- [DrumTranscriber](https://github.com/yoshi-man/DrumTranscriber) - for giving inspiration on using pigeon to label data.
- [pigeonXT](https://github.com/dennisbakhuis/pigeonXT) - for creating an advanced version of pigeon with multi-label capabilities.

We are grateful to the developers and maintainers of these tools for their contributions to the open-source community, which have greatly benefited our project.

## Contact Information

For any questions or inquiries about the HeartsOnFire ADT project, please contact us at skittree@gmail.com.

We hope you find this software useful in your drum scoring endeavors! Happy drumming!
