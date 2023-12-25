import numpy as np
import matplotlib.pyplot as plt

from digi_const import *

plt.figure(figsize=(7,9))

plt.subplot(3, 3, 1)
plt.scatter(np.real(symb_bpsk), np.imag(symb_bpsk))
plt.title('BPSK')

plt.subplot(3, 3, 2)
plt.scatter(np.real(symb_qpsk), np.imag(symb_qpsk))
plt.title('QPSK')

plt.subplot(3, 3, 3)
plt.scatter(np.real(symb_8psk), np.imag(symb_8psk))
plt.title('8PSK')

plt.subplot(3, 3, 4)
plt.scatter(np.real(symb_4qam), np.imag(symb_4qam))
plt.scatter(np.real(symb_qpsk), np.imag(symb_qpsk))
plt.title('$\pi/4$-DQPSK')

plt.subplot(3, 3, 5)
plt.scatter(np.real(symb_qpsk), np.imag(symb_qpsk))
plt.arrow(-1/np.sqrt(2), 1/np.sqrt(2), np.sqrt(2), 0, fc='blue', ec='blue')
plt.arrow(-1/np.sqrt(2),-1/np.sqrt(2), np.sqrt(2), 0, fc='blue', ec='blue')
plt.arrow(-1/np.sqrt(2), 1/np.sqrt(2), 0, -np.sqrt(2), fc='blue', ec='blue')
plt.arrow( 1/np.sqrt(2),-1/np.sqrt(2), 0, np.sqrt(2), fc='blue', ec='blue')

plt.title('MSK')

plt.subplot(3, 3, 6)
plt.scatter(np.real(symb_16qam), np.imag(symb_16qam))
plt.title('16QAM')

plt.subplot(3, 3, 7)
plt.scatter(np.real(symb_64qam), np.imag(symb_64qam))
plt.title('64QAM')

plt.subplot(3, 3, 8)
plt.scatter(np.real(symb_256qam), np.imag(symb_256qam))
plt.title('256QAM')

plt.show()