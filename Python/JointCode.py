#!/usr/bin/env python
# coding: utf-8

# Importing general libraries 
import matplotlib.pyplot as plt
import numpy as np
from commpy.channelcoding import Trellis, conv_encode, turbo_encode, get_ldpc_code_params
import string
# Importing premade functions that will help clean the code
from FuncAndClass import *

for z in range(3):
    print(z)
    # huffman code

    
    sizeOfData = 4**8 # np.random.randint(4, 64)
    symbols = list(string.ascii_uppercase)
    arr = np.random.choice(symbols, sizeOfData) # The code
    
    '''
    arr =  ""
    R = open("DoneImages/data147.txt", "r")
    arr = R.read()
    R.close()   
    '''
    
    strData = ""
    for i in arr:
        strData += i
    origData = binText(strData)
    huffData = huffComp(strData, z)
    freq = calcFreq(strData)  

    # LZWData = LZWEnc(strData)
    # infData = Deflate(strData)
    # infData = LZWData

    sourceCodes = [origData, huffData ]# , LZWData, infData]
    sourceNames = ["Original Data", "Huffman Compression" ]#, "LZW Compression", "Inflate/Deflate Compression"]

    print ("Normally our code would be of size ", len(origData))
    for i in range(0, len(sourceCodes)):
        print ("Using ", sourceNames[i],  len(sourceCodes[i]))
        print ("Compression ratio of " , len(origData)/len(sourceCodes[i]))
        print("")
        # plotting for better visuals
        plt.bar(sourceNames[i], len(sourceCodes[i]), align='center')
        plt.title('Data size comparison')
        plt.savefig('CompSize/Source_Comparison'+str(z)+'.png', format='png')
    plt.show()
    plt.close()
    
    # print(LZWDec(sourceCodes[2]))
    
    
    for i in range(1, len(sourceCodes)):
        # hamming code
        origData = np.array(list(origData),dtype=int)
        JSCData = hammingCoding(sourceCodes[i])

        #LDPCData = get_ldpc_code_params(sourceCodes[i])
        # classTrellis(memory, g_matrix, feedback=0, code_type='default')
        # convData = conv_encode(sourceCodes[i], trellis, termination='term', puncture_matrix=None)
        # turboData = turbo_encode(sourceCodes[i], trellis1, trellis2, interleaver)

        EbNo = np.arange(-20, 10)
        plt.xlabel('EbNo(dB)')
        plt.ylabel('BER')
        plt.title('BER vs SNR')
        plt.yscale('log')
        plt.grid(True)

        sourceCodes[i] = [int(i) for i in sourceCodes[i]]
        monteTransmit(EbNo, origData, strData)
        recieveArr = monteTransmit(EbNo, JSCData, strData, freq, 1)
        
        # recieveArr = monteTransmit(EbNo, dict(subString.split("=") for subString in sourceCodes[i].split(";")) + LDPCData, LDPCData, 2)
        # recieveArr = monteTransmit(EbNo, convData, sourceCodes[i], 3)
        # recieveArr = monteTransmit(EbNo, turboData, sourceCodes[i], 4)
    plt.legend()
    plt.savefig('BERSNR/BERSNR_Comparison'+str(z)+'.png', format='png')
    plt.show()
    plt.close()
    print("")
    

# In[ ]:




