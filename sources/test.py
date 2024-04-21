import numpy as np
import mixer
import preprocessing
import sklearn.decomposition as dec
import librosa
import soundfile as sf
import ica

source1, sample_rate2 = librosa.load("./data/input_data/hello.mp3")
source2, sample_rate1 = librosa.load("./data/input_data/tryvoha.mp3")
source3, sample_rate1 = librosa.load("./data/input_data/vidbiy.mp3")

# myMixer = mixer.Mixer(2, [[4.5, 4.5, 1.0], [6.5, 6.5, 1.0]], [[5.0, 5.0, 5.0], [9.0, 9.0, 6.0]])
# myMixer.create_room(sample_rate1, dimensions=[10.0, 10.0, 10.0], rt = 0.3)
# myMixer.load_sources_from_arrays([source1, source2 * 1.5])

# results = myMixer.simulate_and_return_recordings()

mixed_sources = preprocessing.preprocess_data([source1, source2, source3], [[0.2, 0.4, 0.4], [0.15, 0.7, 0.15], [0.1, 0.2, 0.7]])

EPS = 1e-15

sources = ica.ica(mixed_sources, epsilon=EPS)

transformer = dec.FastICA(n_components = 3)
X_transformed = transformer.fit_transform(list(zip(*mixed_sources))).T
print(X_transformed.shape)

for i in range(len(X_transformed)):
    sf.write(f"./data/output_data/micro{i}.mp3", X_transformed[i], sample_rate1)

