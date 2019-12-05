#SN比を求める

import numpy as np
import librosa
import pyworld
import glob
import csv
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

wav_list = list(pathlib.Path('test/20191202').glob('*.wav'))
SN_list = []
input_list = []

for wav_path in wav_list:
    try:
        input_list.append(wav_path)
        x, sr = librosa.load(wav_path, sr = 16000)
        f0, sp, ap = pyworld.wav2world(x.astype(np.float64), sr)
        arr_sp = np.array(sp)
        
        #２次元の配列を１次元にする
        sum_list = []
        for i in range(len(arr_sp)):
            sum_list.append(sum(arr_sp[i]))
        
        #f0によってspを分類
        voiced_list = []
        noise_list = []
        for i in range(len(f0)):
            if f0[i] == 0.:
                noise_list.append(sum_list[i])
            else:
                voiced_list.append(sum_list[i])
        
        #それぞれの中央値を求める
        Nm = np.median(noise_list)
        Sm = np.median(voiced_list)
        
        #計算
        sn_ratio = 10 * np.log10(Sm / Nm)
        SN_list.append(sn_ratio)
    except ValueError:
        print(wav_path)
        SN_list.append(np.NaN)
        pass

sid = pd.Series(input_list)
sp = pd.Series(SN_list)
df = pd.DataFrame({'id':sid, 'sp':sp})
#plt.hist(SN_list, bins = 30, color='gray', rwidth=.8)
#plt.savefig("SN (2019-11-25 ~ 2019-12-01).png") #pngファイルとして保存する
df.to_csv("SN_2019-11-25-2019-12-01.csv")
