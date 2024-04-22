import numpy as np

import preprocessing
import sklearn.decomposition as dec
import librosa
import soundfile as sf
import ica

source1, sample_rate2 = librosa.load("./data/input_data/hello.mp3")
source2, sample_rate1 = librosa.load("./data/input_data/dunkel.mp3")
source3, sample_rate1 = librosa.load("./data/input_data/pespatron.mp3")

mixed_sources = preprocessing.preprocess_data([source1, source2, source3], [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]])

EPS = 1e-15

sources = ica.ica(mixed_sources, epsilon=EPS)

transformer = dec.FastICA(n_components = 3)
X_transformed = transformer.fit_transform(list(zip(*mixed_sources))).T
# print(X_transformed.shape)

for i in range(len(X_transformed)):
    sf.write(f"./data/output_data/micro{i}.mp3", X_transformed[i], sample_rate1)


