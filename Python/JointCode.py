#%%
# Importing general libraries 
import numpy as np
import matplotlib.pyplot as plt
import random
# Importing the tools I need from the commPy library
from commpy.utilities import hamming_dist
from commpy.channels import awgn
from commpy.modulation import PSKModem, Modem

# Importing premade functions that will help clean the code
from huffman import *
from hamming import *

# %%
# huffman code

string = 'BCCADDBBCC' #The code
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

# %%

# variables we will need when we transmit
SigNoiseR = 20
BER = np.empty([], dtype=float)

#%%

# simulating data transmission over a channel
transArr = np.array(origData, dtype=int)

# Stimulate error in transmission by adding gaussiaan noise
mod = PSKModem(transArr.size)
BER = np.empty([], dtype=float)

# use the monte carlo method to make sure that our code works
modArr = mod.modulate(transArr)
recieveArr = awgn(modArr, SigNoiseR, rate=1.0)
demodArr = mod.demodulate(recieveArr, 'hard')
errors = np.setdiff1d(transArr, demodArr)
 
BER = np.append(BER, 1.0 * errors.size)
print("The number of errors in our code is ", BER)
print("Data Recieved is ", demodArr)

#plt.plot(transArr, BER, 'bo', transArr, BER, 'k')

#%%

# simulating data transmission over a channel
transArr = np.array(arr, dtype=int)

# Stimulate error in transmission by adding gaussiaan noise
mod = PSKModem(transArr.size)
BER = np.empty([], dtype=float)

# use the monte carlo method to make sure that our code works
modArr = mod.modulate(transArr)
recieveArr = awgn(modArr, SigNoiseR, rate=1.0)
demodArr = mod.demodulate(recieveArr, 'hard')
errors = np.setdiff1d(transArr, demodArr)

BER = np.append(BER, 1.0 * errors.size)
print("The number of errors in our code is ", BER)
print("Data Recieved is ", demodArr)

# %%
#turning everything to int and finding the hamming distance (position of error)

transArr = transArr.astype(int)
modArr = modArr.astype(int)
recieveArr = recieveArr.astype(int)
demodArr = demodArr.astype(int)

print(transArr)
print(modArr)
print(recieveArr)
print(demodArr)

# plotting for better visuals
plt.show()

#correction = hamming_dist(transArr, demodArr)
# %%
# printing out everything and finiding the error

print("Original Data is", origData)
print("Data after compressions is", compressedData)
print("Transmitted data is", transArr)
print("Recieved data is", recieveArr)
#print("The hamming distance is " + str(correction))
print("")

# %%
