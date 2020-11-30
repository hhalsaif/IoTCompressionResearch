#%%
# Importing general libraries 
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, randn 
# Importing the tools I need from the commPy library
from commpy.utilities import hamming_dist
from commpy.channels import awgn
from commpy.modulation import PSKModem, Modem

# Importing premade functions that will help clean the code
from huffman import *
from hamming import *
from Transmission import *

# %%

def transmit(transArr, SNR):
    #Simulating data transmission over a channel
    mod = PSKModem(transArr.size)
    # Stimulate error in transmission by adding gaussian noise
    modArr = mod.modulate(transArr)
    recieveArr = awgn(modArr, SNR, rate=1.0)
    demodArr = mod.demodulate(recieveArr, 'hard')

    print("The transferred size is ", transArr.size)
    print("The recieved size is ", demodArr.size)
    #calculating the BER
    errors = (transArr != demodArr).sum()
    BER = errors/demodArr.size
    #Printing the results
    print("The number of errors in our code is ", errors)
    print("Data Recieved is ", demodArr)
    print("The Bit error ratio is ", BER)
    
    #plotting our result
    EbN0arr = range(0,SNR)
    BERarr = [None] * len(EbN0arr)
    N = 500000 * 10
    for n in range (0, len(BERarr)): 
        EbNodB = EbN0arr[n]   
        EbNo=10.0**(EbNodB/10.0)
        x = 2 * (rand(N) >= 0.5) - 1
        noise_std = 1/np.sqrt(2*EbNo)
        y = x + noise_std * randn(N)
        y_d = 2 * (y >= 0) - 1
        error = (x != y_d).sum()
        BERarr[n] = 1.0 * error / N
    plt.semilogy(EbN0arr, BERarr)
    plt.xlabel('EbNo(dB)')
    plt.ylabel('BER')
    plt.title('BER vs SNR')
    
    print("")
    return recieveArr

# %%
# huffman code

string = 'BBCCA' # The code
print ("Our code is ", string)
print()

freq = {}
# Calculating frequency
for c in string:
    if c in freq:
        freq[c] += 1
    else:
        freq[c] = 1

freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
nodes = freq

print(freq)
print()

while len(nodes) > 1:
    (key1, c1) = nodes[-1]
    (key2, c2) = nodes[-2]
    nodes = nodes[:-2]
    node = NodeTree(key1, key2)
    nodes.append((node, c1 + c2))
    nodes = sorted(nodes, key=lambda x: x[1], reverse=True) 

huffmanCode = huffman_code_tree(nodes[0][0])
print(' Char | Huffman code ')
print('----------------------')
for (char, frequency) in freq:
    print(' %-4r |%12s' % (char, huffmanCode[char]))

compdata = ''
for char in string:
   compdata += huffmanCode[char]
 
# %%

# Comparing our compressed code to the normal ASCII code values
valueOfA = '01000001'
valueOfB = '01000010'
valueOfC = '01000011'
valueOfD = '01000100'
origData = ''

for char in string:
    if char == 'A':
        origData += valueOfA
    elif char == 'B':
        origData += valueOfB
    elif char == 'C':
        origData += valueOfC
    elif char == 'D':
        origData += valueOfD

origData = np.array(list(origData), dtype=int)
compressedData = np.array(list(compdata), dtype=int)

print ("Normally our code would be of size ", origData.size)
print ("After compression our code would be of size", compressedData.size)
print ("Compression ratio is", origData.size/compressedData.size)

# %%
# plotting for better visuals
plt.bar('Original Data', origData.size, align='center')
plt.bar('Compressed Data', compressedData.size, align='center')
plt.title('Data size before and after compression')
plt.savefig('HuffmanCode_Comparision.png')
plt.show()

# %%
# hamming code

# Enter the data to be transmitted
data = compdata

# Calculate the no of Redundant Bits Required
m = len(data)
r = calcRedundantBits(m)
print("m = ", m)
print("r = ", r)
# Determine the positions of Redundant Bits
arr = posRedundantBits(data, r)

# Determine the parity bits
arr = calcParityBits(arr, r)

# Data to be transferred
arr = np.array(list(arr), dtype=int)
print("Data transferred is ", arr)

#%%

SNR = 10
transmit(origData, SNR)
recieveArr = transmit(arr, SNR)
plt.show()

# %%
# printing out everything and finiding the error

print("Original Data is", origData)
print("Data after compressions is", compressedData)
print("Transmitted data is", arr)
print("Recieved data is", recieveArr)
print("")

# %%
