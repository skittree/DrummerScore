# DrummerScore - Automatic Drum Transcription (ADT)

DrummerScore is a project aimed at creating an Automatic Drum Transcription (ADT) system to help beginner drummers create drum scores.

## Stack
### Server
- FastAPI
- Jinja2
- TailwindCSS + DaisyUI
- JavaScript
- HTMX

### Notebook
- Jupyter
- Pandas
- Torch
- Librosa

## Images

This project uses __spectrogram visualization__ to help streamline the scoring process.

![DrummerScore Interface](https://github.com/skittree/DrummerScore/blob/master/notebooks/figures/roll.png)

There is a distinct lack of visualization methods in available scoring software, which is especially useful for drum transcription, even without predictions. It can help us correct any mistakes the model made placing the drums on the drum machine:

![DrummerScore Drum Machine with Model Predictions](https://github.com/skittree/DrummerScore/blob/master/notebooks/figures/drummachine.png)

## Notebooks Included

- `generating our own dataset.ipynb`: This notebook provides instructions and code for generating your own iterative dataset for training the ADT model.
- `model training.ipynb`: This notebook guides you through the process of training the ADT model using the generated dataset.
- `main.ipynb`: This notebook allows you to input an .mp3 file and use the trained ADT model to receive a labeled MIDI file for the source-separated drum track.

## Contributions

We welcome contributions to the DrummerScore project! If you would like to contribute, please open an issue or submit a pull request on GitHub. We appreciate your feedback and support in making this project better.

## License

This project is released under the [MIT License](https://opensource.org/license/mit/), which means it is open-source and free to use, modify, and distribute for personal and commercial purposes.

## Acknowledgements

I would like to express my gratitude to the following resources, libraries, and tools that have been used in this project:

- [Librosa](https://librosa.org/doc/main/index.html) - for audio processing and feature extraction.
- [pretty_midi](https://craffel.github.io/pretty-midi/) - for creating MIDI files in Python.
- [demucs](https://github.com/facebookresearch/demucs) - for source separating drum tracks from input .mp3 files.
- [torch_audiomentations](https://github.com/asteroid-team/torch-audiomentations) - for data augmentation during model training.
- [Pandas](https://pandas.pydata.org/) - for data manipulation and analysis in Python.
- [DrumTranscriber](https://github.com/yoshi-man/DrumTranscriber) - for giving inspiration on using pigeon to label data.
- [pigeonXT](https://github.com/dennisbakhuis/pigeonXT) - for creating an advanced version of pigeon with multi-label capabilities.

I am grateful to the developers and maintainers of these tools for their contributions to the open-source community, which have greatly benefited my project.

## Contact Information

For any questions or inquiries about the DrummerScore project, please contact me at skittree@gmail.com.
