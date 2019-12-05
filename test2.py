#SN比を求める

import numpy as np
import librosa
import pyworld

x, sr = librosa.load('*.wav', sr = 16000)
f0, sp, ap = pyworld.wav2world(x.astype(np.float64), sr)

arr_sp = np.array(sp)
sum_list = []
for i in range(len(arr_sp)):
    sum_list.append(sum(arr_sp[i]))

voiced_list = []
noise_list = []
for i in range(len(f0)):
    if f0[i] == 0.:
        noise_list.append(sum_list[i])
    else:
        voiced_list.append(sum_list[i])

Nm = np.median(noise_list)
Sm = np.median(voiced_list)

SN_ratio = 10 * np.log10(Sm / Nm)
print('SN_ratio:', SN_ratio)
