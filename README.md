# RealtimeAudioPython
Contains example of real-time audio applications developed in Python 3.6
We encorage re-use of this code. Please cite the following source literature: De Pra Y., Fontana F.: Programming Real-Time Sound in Python. Applied Sciences. 2020; 10(12):4214. https://doi.org/10.3390/app10124214

- low_filter_raw.py apply a low pass filter to a wave file given as argument. The user interface shows the FFT of the signal. A cursor allows to change the filter cutoff frequency. Example usage: python low_filter_raw.py --input=./rock.wav 

- python butt_filters.py create low, high or bandpass filters depending on the type argument. The user interface shows the FFT of the signal. A cursor allows to change the filter parameters. Example usage: python butt_filters.py --input=./rock.wav --type=high|low|noparam=bandpass 

- Ring folder contains the files to test the Diode Ring Modulator implementation

- VCF folder contains the files to test the VCS3 algorithm

- Sine folder contains the files to test the Sine oscillator

- rock.wav is a test waveform.
