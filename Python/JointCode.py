#%%
# Importing general libraries 
import numpy as np
import matplotlib.pyplot as plt
import random
# Importing the tools I need from the commPy library
from commpy.utilities import hamming_dist
from commpy.channels import awgn

# Importing premade functions that will help with celaning the code
from huffman import *
from hamming import *

# %%
# huffman code

#The code
string = 'BCCADDBC'
print ("Our code is ", string)
print()

# Calculating frequency
freq = {}
for c in string:
    if c in freq:
        freq[c] += 1
    else:
        freq[c] = 1

freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
nodes = freq

while len(nodes) > 1:
    (key1, c1) = nodes[-1]
    (key2, c2) = nodes[-2]
    nodes = nodes[:-2]
    node = NodeTree(key1, key2)
    nodes.append((node, c1 + c2))
    nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

huffmanCode = huffman_code_tree(nodes[0][0])

print(freq)
print()
print(' Char | Huffman code ')
print('----------------------')
for (char, frequency) in freq:
    print(' %-4r |%12s' % (char, huffmanCode[char]))

transData = ''
for char in string:
    transData += huffmanCode[char]

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
compressedData = np.array(list(transData), dtype=int)

print ("Normally our code would be of size ", origData.size)
print ("After compression our code would be of size", compressedData.size)
print ("Compression ratio is", origData.size/compressedData.size)

# %%
# plotting for better visuals
plt.barh('Original Data', origData.size, align='center')
plt.barh('Compressed Data', compressedData.size, align='center')
plt.title('Data size before and after compression')
plt.show()

# %%
# hamming code

# Enter the data to be transmitted
data = transData

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
# Stimulate error in transmission by adding gaussiaan noise
transArr = np.array(arr, dtype=float)

SigNoiseR = 40 # generate SNR -20, 10

recieveArr = awgn(np.array(transArr), SigNoiseR, rate=1.0)

print("Data Recieved is ", recieveArr)

# %%
# plotting for better visuals
plt.plot(transArr, label="Transmitted Array")
plt.plot(recieveArr, label="Recieved Array")
plt.legend()
plt.show()

# %%
#turning everything to int and finding the hamming distance (position of error)
transArr = transArr.astype(int)
recieveArr = recieveArr.astype(int)
correction = hamming_dist(transArr, recieveArr)

transData = np.array(list(transData), dtype=int)

# %%
# printing out everything and finiding the error

print("Original Data is", origData)
print("Data after compressions is", transData)
print("Transmitted data is", transArr)
print("Recieved data is", recieveArr)
print("The hamming distance is " + str(correction))
print("")

# %%
