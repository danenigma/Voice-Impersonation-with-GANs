import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment
import tempfile
import os

def read_spect_matrix(audio_file_list):
	
	data = []
	
	for file in audio_file_list:
		
		chunk = AudioSegment.from_file(file, "mp4")
		new_file, filename = tempfile.mkstemp()
		chunk.export(filename, format="wav")
		sample_rate, samples = wavfile.read(filename)
		frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
		data.append(spectrogram)
		os.close(new_file)
		
	return data
	
if __name__ == '__main__':

	audio_file_list = ['../data/00001.m4a']*10	
	data = read_spect_matrix(audio_file_list)
	
	'''
	plt.imshow(data[0])
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.show()
	'''
