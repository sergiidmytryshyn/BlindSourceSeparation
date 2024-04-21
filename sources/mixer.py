# Mixer class
import pyroomacoustics as pra
import librosa
import numpy as np

class Mixer:

    def __init__(self, num_of_sources: int, positions_of_microphones: list[list[float]], positions_of_sources: list[list[float]]):

        self.num_of_sources = num_of_sources
        
        if len(positions_of_microphones) == self.num_of_sources and len(positions_of_sources) == num_of_sources:
            self.positions_of_microphones = positions_of_microphones
            self.positions_of_source = positions_of_sources
        else:
            raise ValueError("Number of sources must be equal to number of microphones")

        self.room = None

    def create_room(self, Fs, dimensions = [10.0, 10.0, 10.0], rt = 0.1):
        room_dimension = dimensions
        reverbation_time = rt
        
        e_absorption, max_order = pra.inverse_sabine(reverbation_time, room_dimension)

        self.room = pra.ShoeBox(room_dimension, fs=Fs, materials=pra.Material(e_absorption), max_order=max_order)

        mic_positions = np.array(self.positions_of_microphones).transpose()
        self.room.add_microphone_array(mic_positions)

    def load_sources_from_files(self, file_names = list[str]):
        if len(file_names) == self.num_of_sources:

            for ind, name in enumerate(file_names):
                source, _ = librosa.load(name)
                self.room.add_source(self.positions_of_source[ind], signal = source, delay = 0)

        else:
            raise ValueError("Number of files is not equal to number of sources specified!")
    
    def load_sources_from_arrays(self, source_signals):
        if len(source_signals) == self.num_of_sources:

            for ind, source in enumerate(source_signals):
                self.room.add_source(self.positions_of_source[ind], signal = source, delay = 0)

        else:
            raise ValueError("Number of sources is not equal to expected number of sources specified earlier!")
    
    def simulate_and_return_recordings(self):
        self.room.simulate()
        return self.room.mic_array.signals