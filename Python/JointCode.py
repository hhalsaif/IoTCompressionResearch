#%%
# Importing general libraries 
import matplotlib.pyplot as plt
import numpy as np
import string
# Importing the tools I need from the commPy library
from commpy.utilities import hamming_dist
from commpy.channels import awgn
from commpy.modulation import PSKModem, Modem

# Importing premade functions that will help clean the code
from huffman import *
from hamming import *

# %%
# huffman code
sizeOfData = 50000
symbols = list(string.ascii_uppercase)
arr = np.random.choice(symbols, sizeOfData) # The code
string = ""
for i in arr:
    string += i
#print("Our code is ", string)
print("")

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
origData = ''.join(format(ord(i), 'b') for i in string)
origData = np.array(list(origData), dtype=int)
compressedData = np.array(list(compdata),dtype=int)

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
plt.close()

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

# %%

def transmit(transArr, SNR):
    numErrs = 0
    print(transArr.size)
    #Simulating data transmission over a channel
    mod = PSKModem(transArr.size)
    # Stimulate error in transmission by adding gaussian noise
    modArr = mod.modulate(transArr)
    recieveArr = awgn(modArr, SNR, rate=2)
    demodArr = mod.demodulate(recieveArr, 'hard')
    #calculating the BER
    print(transArr.size, " and ", demodArr.size)
    numErrs += np.sum(transArr != demodArr)
    BER = numErrs/demodArr.size
    #Plotting and Printing the results
    plt.semilogy(np.flip(np.arange(SNR)), np.linspace(0, BER, SNR), label = transArr.size)
    print("The number of errors in our code is ", numErrs)
    print("Data Transmited is ", transArr)
    print("Data Recieved is ", demodArr)
    print("The Bit error ratio is ", BER)
    print("")
    return demodArr

def monteTransmit(EbNo, transArr):
    print(transArr.size)
    BERarr = [None] * EbNo.size
    for i in EbNo:
        SNR = EbNo[i]
        #reset the bit counters
        numErrs = 0
        #Simulating data transmission over a channel
        mod = PSKModem(transArr.size)
        # Stimulate error in transmission by adding gaussian noise
        modArr = mod.modulate(transArr)
        recieveArr = awgn(modArr, SNR, rate=2)
        demodArr = mod.demodulate(recieveArr, 'hard')
        #calculating the BER
        numErrs = np.sum(transArr != demodArr)
        BERarr[i] = numErrs/demodArr.size
    plt.semilogy(EbNo, BERarr, label = transArr.size)
    print("The number of errors in our code is ", numErrs)
    print("Data Transmited is ", transArr)
    print("Data Recieved is ", demodArr)
    print("The Bit error ratio is ", BERarr[i])
    print("")  
    return demodArr     

#%%

SNR = 10
plt.xlabel('EbNo(dB)')
plt.ylabel('BER')
plt.title('BER vs SNR')
plt.yscale('log')
plt.axis([0, 10, 1e-6, 0.1])
plt.grid(True)
transmit(origData, SNR)
recieveArr = transmit(arr, SNR)
plt.legend()
plt.savefig('BERSNR_Comparision.png')
plt.show()
plt.close()

"""
EbNo = np.arange(5)
plt.xlabel('EbNo(dB)')
plt.ylabel('BER')
plt.title('BER vs SNR')
plt.yscale('log')
plt.grid(True)
monteTransmit(EbNo, origData)
monteTransmit(EbNo, arr)
plt.legend()
plt.savefig('BERSNR_Comparision2.png')
plt.show()
plt.close()
"""
# %%
# printing out everything and finiding the error

print("Original Data is", origData)
print("Data after compressions is", compressedData)
print("Transmitted data is", arr)
print("Recieved data is", recieveArr)
print("")

# %%
