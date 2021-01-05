#%%
# Importing general libraries 
import matplotlib.pyplot as plt
import numpy as np
import string
# Importing premade functions that will help clean the code
from FuncAndClass import *

# %%
# huffman code
sizeOfData = 100000 #np.random.randint(10000,50000)
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

#%%

EbNo = np.arange(0,20)
plt.xlabel('EbNo(dB)')
plt.ylabel('BER')
plt.title('BER vs SNR')
plt.yscale('log')
plt.grid(True)
monteTransmit(EbNo, origData)
recieveArr = monteTransmit(EbNo, arr)
plt.legend()
plt.savefig('BERSNR_Comparision.png')
plt.show()
plt.close()


# %%
# printing out everything and finiding the error

print("Original Data is", origData)
print("Data after compressions is", compressedData)
print("Transmitted data is", arr)
print("Recieved data is", recieveArr)
print("")

# %%
