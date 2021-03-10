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
    
    '''
    sizeOfData = np.random.randint(30,100)
    symbols = list(string.ascii_uppercase)
    arr = np.random.choice(symbols, sizeOfData) # The code
    '''

    arr =  ""
    R = open("DoneImages/data147.txt", "r")
    arr = R.read()
    R.close()   
    compData = huffComp(arr, z)
    origData = binText(arr)

    print("")
    print ("Normally our code would be of size ", len(origData))
    print ("After compression our code would be of size",  len(compData))
    print ("Compression ratio is", len(origData)/len(compData))
    print("")

    # plotting for better visuals
    plt.bar('Original Data', len(origData), align='center')
    plt.bar('Compressed Data', len(compData), align='center')
    plt.title('Data size before and after compression')
    plt.savefig('HuffComp/HuffmanCode_Comparision'+str(z)+'.png', format='png')
    plt.show()
    plt.close()


    # hamming code
    JSCData = hammingCoding(compData)
    convData = compData.conv_encode()
    turboData = turbo_encode(compData, trellis1, trellis2, interleaver)
    LDPCData = get_ldpc_code_params(compData)

    EbNo = np.arange(-10, 20)
    plt.xlabel('EbNo(dB)')
    plt.ylabel('BER')
    plt.title('BER vs SNR')
    plt.yscale('log')
    plt.grid(True)

    monteTransmit(EbNo, np.array(list(origData),dtype=int))
    # monteTransmit(EbNo, correctedData)
    recieveArr = monteTransmit(EbNo, JSCData, compData, 1)
    recieveArr = monteTransmit(EbNo, convData, compData, 2)
    recieveArr = monteTransmit(EbNo, turboData, compData, 3)
    recieveArr = monteTransmit(EbNo, LDPCData, compData, 4)


    plt.legend()
    plt.savefig('BERSNR/BERSNR_Comparision'+str(z)+'.png', format='png')
    plt.show()
    plt.close()


    # printing out everything and finiding the error

    print("Original Data is", origData)
    print("Data after compressions is", compData)
    print("Transmitted data is", JSCData)
    print("Recieved data is", recieveArr)
    print("")


# In[ ]:




