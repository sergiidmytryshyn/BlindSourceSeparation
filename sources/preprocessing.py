import numpy as np

def mix_sound(sounds, mixture):
    mix_sound = np.zeros(sounds[0].shape[0])
    for i in range(len(mixture)):
        mix_sound += sounds[i]*mixture[i]
    return mix_sound
        


def preprocess_data(sounds, mixtures):
    min_length = float("inf")
    for sound in sounds:
        length = sound.shape[0]
        if length < min_length:
            min_length = length

    cut_sounds = []
    for sound in sounds:
        cut_sounds.append(sound[:min_length])

    mixed_sounds = []
    for i in range(len(sounds)):
        mixed_sounds.append(mix_sound(cut_sounds,mixtures[i]))
    return mixed_sounds
    

    
