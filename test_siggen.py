import numpy as np
import scipy.signal as signal
from digi_const import *
from commpy import filters
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

symbols = np.random.choice(np.array(symb_16qam.flatten()), (65536*4,))

P_upsample = 7
N_downsample = 3

(t,h) = filters.rcosfilter(P_upsample*10, 0.5, 1, P_upsample)
upsymbols = signal.upfirdn(np.array(h), symbols, P_upsample, N_downsample)
fc = 0.1
sig = upsymbols * np.exp(2j*fc*np.pi*np.arange(len(upsymbols)))


plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.plot(np.real(upsymbols[1:100]))
plt.plot(np.imag(upsymbols[1:100]))
plt.title("前100点")

plt.subplot(1,3,2)
plt.plot(t, h,'-o')
plt.title("成型滤波器（时域）")

plt.subplot(1,3,3)
f, Pxx_den = signal.welch(sig[100:], 1, nperseg=1024, return_onesided=False)
f = np.fft.fftshift(f)
Pxx_den = np.fft.fftshift(Pxx_den)
plt.semilogy(f, Pxx_den)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title("功率谱")
plt.grid()
plt.show()