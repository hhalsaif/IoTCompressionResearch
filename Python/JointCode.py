#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# Importing general libraries 
import matplotlib.pyplot as plt
import numpy as np
import string
# Importing premade functions that will help clean the code
from FuncAndClass import *


# In[2]:

for z in range(50):
    # huffman code
    sizeOfData = 30000 #np.random.randint(10000,50000)
    symbols = list(string.ascii_uppercase)
    arr = np.random.choice(symbols, sizeOfData) # The code

    string = ""
    for i in arr:
        string += i
    print("")
    f = open("string.txt", 'w')
    f.write('string = ' + string)
    f.close()

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


    # In[3]:


    # Comparing our compressed code to the normal ASCII code values
    origData = ''.join(format(ord(i), 'b') for i in string)
    originalData = np.array(list(origData), dtype=int)
    compressedData = np.array(list(compdata),dtype=int)

    print("")
    print ("Normally our code would be of size ", originalData.size)
    print ("After compression our code would be of size", compressedData.size)
    print ("Compression ratio is", originalData.size/compressedData.size)
    print("")


    # In[4]:


    # plotting for better visuals
    plt.bar('Original Data', originalData.size, align='center')
    plt.bar('Compressed Data', compressedData.size, align='center')
    plt.title('Data size before and after compression')
    plt.savefig('HuffComp/HuffmanCode_Comparision'+str(z)+'.png', format='png')
    plt.show()
    plt.close()


    # In[5]:


    # hamming code
    JSCData = hammingCoding(compdata)
    correctedData = hammingCoding(origData)


    # In[6]:


    EbNo = np.arange(-5,25)
    plt.xlabel('EbNo(dB)')
    plt.ylabel('BER')
    plt.title('BER vs SNR')
    plt.yscale('log')
    plt.grid(True)
    monteTransmit(EbNo, originalData)
    monteTransmit(EbNo, compressedData)
    monteTransmit(EbNo, correctedData)
    recieveArr = monteTransmit(EbNo, JSCData)
    plt.legend()
    plt.savefig('BERSNR/BERSNR_Comparision'+str(z)+'.png', format='png')
    plt.show()
    plt.close()


    # In[ ]:


    # printing out everything and finiding the error

    print("Original Data is", originalData)
    print("Data after compressions is", compressedData)
    print("Transmitted data is", JSCData)
    print("Recieved data is", recieveArr)
    print("")


# In[ ]:




