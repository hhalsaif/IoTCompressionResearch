#!/usr/bin/env python
# coding: utf-8

# Importing general libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from commpy.channelcoding import Trellis, conv_encode, turbo_encode, get_ldpc_code_params
import string
# Importing premade functions that will help clean the code
from FuncAndClass import *
# Importing tools for benchmarking
from time import perf_counter_ns
from pypapi import papi_high

for z in range(1):
    print(z)
    # Data Generation of random string
    sizeOfData = 4*12 #np.random.randint(1, 64)
    symbols = list(string.ascii_uppercase)
    arr = np.random.choice(symbols, sizeOfData) # The code
    strData = ''.join([str(i) for i in arr])
    
    # Data generation of random number from the dataset
    df = pd.read_csv("Datasets/IOT-temp.csv")    
    temps = pd.Series(df['temp'])
    dataNo = np.random.randint(1,97600)
    data = temps[dataNo]
    print(data)
    byteData = data.tobytes()
    # byteData = bytes(data, 'utf-8')

    #How many compression techniques do we want to use 
    noOfData = 6   

    # Data that will help with comparisions of compression techniques
    sourceNames = ["Original", "Huffman", "LZW", "DEFLATE", "LZMA", "Zstandard"]
    sourceCodes=[0] * noOfData 
    decodedCodes=[0] * noOfData
    compAlgo = [binIt, huffComp, LZWEnc, deflate, LZMAComp, zstdComp] # Functinos for the compression techniques we use
    deCompAlgo = [returnIt, huffDecomp, LZWDec, inflate, LZMADeComp, zstdDeComp] # Functions for decompression techniques we use

    # Tracking for benchmarking
    timeEn = [0.0] * noOfData ; timeDe = [0.0] * noOfData ; flopsEn = [0.0] * noOfData; flopsDe = [0.0] * noOfData

    for i in range(noOfData):
        # Time taken to compress
        timeStart = perf_counter_ns()
        sourceCodes[i] = compAlgo[i](byteData)
        timeStop = perf_counter_ns()
        timeEn[i] = timeStop - timeStart

        '''
        #Number of Flops 
        papi_high.Flops() 
        result = papi_high.flops(rtime=0,ptime=0,flpops=0,mflops=0)
        flopsEn[i] = result.mflops
        papi_high.stop_counters()
       '''

    for i in range(noOfData):     
        # Time taken to decompress
        timeStart = perf_counter_ns()
        decodedCodes[i] = deCompAlgo[i](sourceCodes[i])
        timeStop = perf_counter_ns()
        timeDe[i] = timeStop - timeStart
        
        '''
        #Number of Flops 
        papi_high.flops()
        decodedCodes[i] = deCompAlgo[i](sourceCodes[i])
        result = papi_high.flops()
        flopsDe[i] = result.mflops
        papi_high.stop_counters()
        '''    
    print("Normally our code would be of size ", len(sourceCodes[0])) ; print("")

    for i in range(0, noOfData):
        print("Using ", sourceNames[i],  len(sourceCodes[i]))
        print("Compression ratio of " , len(sourceCodes[0])/len(sourceCodes[i]))
        print("The time taken to compress", timeEn[i])
        print("The time taken to decompress", timeDe[i])
        print("Number of FLOPS to compress", flopsEn[i])
        print("Number of FLOPS to decompress", flopsDe[i])
        print("")
        # plotting for better visuals
        plt.bar(sourceNames[i], len(sourceCodes[i]), align='center')
        plt.title('Data size comparison')
        plt.savefig('CompSize/Source_Comparison'+str(z)+'.png', format='png')
    plt.show()
    plt.close()

    sourceCodes.pop(2)
    sourceNames.pop(2)
    for i in range(1, noOfData):
        # hamming code
        origData = np.array(list(sourceCodes[0]),dtype=int)
        hammData = hammingCoding(sourceCodes[i])
        # LDPCData = get_ldpc_code_params(sourceCodes[i])
        # classTrellis(memory, g_matrix, feedback=0, code_type='default')
        # convData = conv_encode(sourceCodes[i], trellis, termination='term', puncture_matrix=None)
        # turboData = turbo_encode(sourceCodes[i], trellis1, trellis2, interleaver)

        EbNo = np.arange(-5, 5)
        plt.xlabel('EbNo(dB)')
        plt.ylabel('BER')
        plt.title('BER vs SNR')
        plt.yscale('log')
        plt.grid(True)

        monteTransmit(EbNo, origData, byteData)
        recieveArr = monteTransmit(EbNo, hammData, byteData, 1, i)
        # recieveArr = monteTransmit(EbNo, dict(subString.split("=") for subString in sourceCodes[i].split(";")) + LDPCData, LDPCData, 2)
        # recieveArr = monteTransmit(EbNo, convData, sourceCodes[i], 3)
        # recieveArr = monteTransmit(EbNo, turboData, sourceCodes[i], 4)
        plt.legend()
        plt.savefig('BERSNR/' + sourceNames[i] + '/BERSNR_Comparison'+str(z)+'.png', format='png')
        plt.show()
        plt.close()
        print("")
    
        