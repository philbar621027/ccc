import numpy as np
from commpy import filters
from digi_const import *
from scipy import signal
import matplotlib.pyplot as plt

def mod05(x):
    mod1 = np.mod(x, 1)
    if(mod1>0.5):
        return (1-mod1)*np.sign(x)
    else:
        return mod1*np.sign(x)
    
def calc_fft_mom(insig, nk, mk): #使用fft计算高阶矩的估计值
    N = len(insig)
    mom = (insig ** (nk-mk)) * (np.conj(insig)**mk)
    fft_mom = np.abs(np.fft.fft(mom))/N
    return fft_mom

def get_top_n(arr, n): #函数，得到最多n个点
    max_indices = np.argsort(arr)[::-1]

    indices_set = []
    count = 1
    indices_set.append(max_indices[0])
    
    for i in max_indices:
        flag_found = 0
        for j in indices_set:
            if(np.abs(j-i)<150):
                flag_found = 1
                
        if(flag_found == 0):
            count += 1
            indices_set.append(i)

        if count == n:
            break

    return np.array(indices_set)

def prominance(data, index, n=100): #通过均峰比确定最显著程度
    if len(data) < 2*n:
        return None
    else:
        # 获取相邻n个元素
        prev_n = data[index-n-100:index-100]
        next_n = data[index+100:index+n+100]
        
        # 拼接前后两个数据窗口
        window = np.hstack((prev_n, next_n))
        
        # 按照取得的数据窗口的长度，计算循环下标
        c_index = index % len(window)
        
        # 计平均值
        mean =np.mean(window)
        
        # 计算突出程度
        abs_diff = np.abs(data[index] - mean)
        abs_mean = np.abs(mean)
        return abs_diff / abs_mean

def batch_prominance(data, index, n=100, threshold=7): #对多个显著程度进行排序
    promi = []
    for i in index:
        promi.append(prominance(data, i, n=100))

    # 获取索引的排序结果
    sorted_index = np.argsort(promi)[::-1]
    promi = np.sort(promi)[::-1]
    topn = np.sum(promi>threshold)
    # 返回排序后的索引
    return promi[:topn], index[sorted_index][:topn]

def finetune_cyclic_mom(sig, nk, mk, alpha0, init_bins=17, refine_bins=40, refine_range=0.2):
    N = len(sig)
    frange = np.arange(N)/N - 0.5  # FFT的频率横坐标

    # 初步搜索
    initial_idx = np.argmax(np.abs(np.fft.fftshift(calc_fft_mom(sig, nk, mk))))
    fbins = np.linspace(alpha0-2/N, alpha0 + 2/N, init_bins)
    res = [calc_alpha_mom(sig, nk, mk, alpha).tolist() for alpha in fbins]

    idx_max = np.argmax(abs(np.real(res))-abs(np.imag(res)))
    
    # 细化搜索
    fbins = np.linspace(fbins[idx_max]-refine_range/N, fbins[idx_max] + refine_range/N, refine_bins)
    res = [calc_alpha_mom(sig, nk, mk, alpha).tolist() for alpha in fbins]

    nk = np.argmax(abs(np.real(res))-abs(np.imag(res)))

    cyclic_freq = fbins[nk]
    mom_value = res[nk]

    return cyclic_freq, mom_value

def calc_alpha_mom(insig, nk, mk, alpha): #使用fft计算高阶矩的估计值
    N = len(insig)
    expj = np.exp(-2j*alpha*np.pi*np.arange(N))
    mom = np.mean(expj * (insig ** (nk-mk)) * (np.conj(insig)**mk))
    return mom

def calculate_cyclic_freq(sig, nk, mk, least_prominance=5):
    N = len(sig)
    frange = np.arange(N)/N - 0.5
    fft_mom_shift = np.fft.fftshift(calc_fft_mom(sig, nk, mk))
    cpu_fft_mom_shift = fft_mom_shift

    maxi = get_top_n(cpu_fft_mom_shift, 7)
    promi, idx = batch_prominance(cpu_fft_mom_shift, maxi, 20)

    cycfreq_list = frange[idx[promi>least_prominance]]

    fined_cycfreqs = []
    cyclic_moments = []

    for alpha0 in cycfreq_list:
        alpha_fined, cyc_mom = finetune_cyclic_mom(sig, nk, mk, alpha0,init_bins=17, refine_bins= 50)
        fined_cycfreqs.append(alpha_fined)
        cyclic_moments.append(cyc_mom)

    return fined_cycfreqs, cyclic_moments


# 信号生成 
symbols = np.random.choice(np.array(symb_qpsk), (65536*4,))
P_upsample = 7
N_downsample = 3
(t,h) = filters.rcosfilter(P_upsample*10, 0.5, 1, P_upsample)
upsymbols = signal.upfirdn(np.array(h), symbols, P_upsample, N_downsample)
fc = 0.007
sig = upsymbols * np.exp(2j*fc*np.pi*np.arange(len(upsymbols)))
sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2))

# 使用cc(2,1) 计算符号速率
cycfreqs, cycmoms = calculate_cyclic_freq(sig, 2, 1)

# print("cc(2,1) list:")
# print(cycfreqs)
# print("corresponding c.m.:")
# print(cycmoms)

if(len(cycfreqs)>=2):
    f_sym = np.abs(cycfreqs[1])
else:
    f_sym = -1

mombooks = {}

idx = 0
if(f_sym>0):
    for alpha in cycfreqs: 
        k = np.round(alpha/f_sym)
        mombooks[(2,1,k)] = cycmoms[idx]
        idx = idx + 1
else:
    mombooks[(2,1,0)] = 0

# 使用cc(4,0) 计算载波频率
cycfreqs, cycmoms = calculate_cyclic_freq(sig, 4, 0)

if len(cycfreqs)==0:
    print("cc(4,0) is empty, maybe M-PSK(M>=8)")
else:
    print("cc(4,0) list:")
    print(cycfreqs)
    print("corresponding c.m.:")
    print(cycmoms)

if(len(cycfreqs)>2):
    f_c = np.abs(cycfreqs[0])/4
else:
    f_c = -1

idx = 0
nextk = 0

if(f_c>0):
    mombooks[(4,0,idx)] = cycmoms[idx]
    idx = idx + 1
    for alpha in cycfreqs[1:]:
        k_temp = np.floor((idx+1)/2)
        if(np.abs(alpha - mod05(4*f_c+k_temp*f_sym))<1e-3):
            mombooks[(4,0,k_temp)] = cycmoms[idx]
            idx = idx + 1

        if(np.abs(alpha - mod05(4*f_c-k_temp*f_sym))<1e-3):
            mombooks[(4,0,-k_temp)] = cycmoms[idx]
            idx = idx + 1            
else:
    mombooks[(4,0,0)] = 0

print(mombooks)
    


