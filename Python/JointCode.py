#!/usr/bin/env python
# coding: utf-8

# Importing general libraries 
import matplotlib.pyplot as plt
import numpy as np
from commpy.channelcoding import *
import string
# Importing premade functions that will help clean the code
from FuncAndClass import *


for z in range(3):
    print(z)
    # huffman code

    
    sizeOfData = np.random.randint(10,100)
    symbols = list(string.ascii_uppercase)
    arr = np.random.choice(symbols, sizeOfData) # The code
    
    '''
    arr =  ""
    R = open("DoneImages/data147.txt", "r")
    arr = R.read()
    R.close()   
    '''

    huffData = huffComp(arr, z)
    origData = binText(arr)
    LZWData = LZWEnc(arr)
    infData = Deflate(arr)

    sourceCodes = [origData, huffData, LZWData, infData]
    sourceNames = ["Original Data", "Huffman Compression", "LZW Compression", "Inflate/Deflate Compression"]

    for i in range(0, len(sourceCodes)):
        print("")
        print ("Normally our code would be of size ", len(origData))
        print ("Using ", sourceNames[i],  len(sourceCodes[i]))
        print ("Compression ratio of " , len(origData)/len(sourceCodes[i]))
        print("")
        # plotting for better visuals
        plt.bar(sourceNames[i], len(sourceCodes[i], align='center'))
        plt.title('Data size comparision')
        plt.savefig('CompSize/Source_Comparision'+str(z)+'.png', format='png')
    plt.show()
    plt.close()

    
    '''
     for i in range(0, len(sourceCodes)):
        # hamming code
        JSCData = hammingCoding(comparision)
        LDPCData = get_ldpc_code_params(sourceCodes[i])

        # classTrellis(memory, g_matrix, feedback=0, code_type='default')
        # convData = conv_encode(sourceCodes[i], trellis, termination='term', puncture_matrix=None)
        # turboData = turbo_encode(sourceCodes[i], trellis1, trellis2, interleaver)

        EbNo = np.arange(-10, 20)
        plt.xlabel('EbNo(dB)')
        plt.ylabel('BER')
        plt.title('BER vs SNR')
        plt.yscale('log')
        plt.grid(True)

        monteTransmit(EbNo, np.array(list(origData),dtype=int))
        recieveArr = monteTransmit(EbNo, JSCData, sourceCodes[i], 1)
        recieveArr = monteTransmit(EbNo, dict(subString.split("=") for subString in sourceCodes[i].split(";")) + LDPCData, LDPCData, 2)

        # recieveArr = monteTransmit(EbNo, convData, sourceCodes[i], 3)
        # recieveArr = monteTransmit(EbNo, turboData, sourceCodes[i], 4)

    plt.legend()
    plt.savefig('BERSNR/BERSNR_Comparision'+str(z)+'.png', format='png')
    plt.show()
    plt.close()
    print("")
    '''

# In[ ]:




