#!/usr/bin/env python
# coding: utf-8

# Importing general libraries 
import matplotlib.pyplot as plt
import numpy as np
from commpy.channelcoding import Trellis, conv_encode, turbo_encode, get_ldpc_code_params
import string
# Importing premade functions that will help clean the code
from FuncAndClass import *
# Importing tools for benchmarking
from time import perf_counter_ns
from pypapi import papi_high

for z in range(1):
    print(z)
    # Data Generation
    sizeOfData = 63 #np.random.randint(1, 64)
    symbols = list(string.ascii_uppercase)
    arr = np.random.choice(symbols, sizeOfData) # The code
    strData = ''.join([str(i) for i in arr])
    #byteData = bytes(strData, 'utf-8')

    #How many compression techniques do we want to use 
    noOfData = 6   

    # Data that will help with comparisions of compression techniques
    sourceNames = ["Original Data", "Huffman Compression", "LZW Compression", "DEFLATE Compression", "LZMA Compression", "Zstandard Compression"]
    sourceCodes=[0] * noOfData 
    decodedCodes=[0] * noOfData
    compAlgo = [binText, huffComp, LZWEnc, deflate, LZMAComp, zstdComp] # Functinos of the compression techniques we use
    deCompAlgo = [textBin, huffDecomp, LZWDec, inflate, LZMADeComp, zstdDeComp] # Functions of decompression techniques we use

    # Tracking for benchmarking
    timeEn = [0.0] * noOfData ; timeDe = [0.0] * noOfData ; flopsEn = [0.0] * noOfData; flopsDe = [0.0] * noOfData

    for i in range(noOfData):
        # Time taken to compress
        timeStart = perf_counter_ns()
        if i == 0: sourceCodes[i] = compAlgo[i](strData)
        else:sourceCodes[i] = compAlgo[i](strData)
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
    for i in range(noOfData): 
        if strData != decodedCodes[i]: print(str(i) + "\n" + "Error in decoding found" + "\n" + "original data: " + strData + "\n" + "decoded data: " + decodedCodes[i])
    
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

    '''
    for i in range(1, noOfData):
        # hamming code
        origData = np.array(list(sourceCodes[0]),dtype=int)
        # hammData = hammingCoding(sourceCodes[i])
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

        monteTransmit(EbNo, origData, strData)
        # recieveArr = monteTransmit(EbNo, hammData, strData, 1, i)
        # recieveArr = monteTransmit(EbNo, dict(subString.split("=") for subString in sourceCodes[i].split(";")) + LDPCData, LDPCData, 2)
        # recieveArr = monteTransmit(EbNo, convData, sourceCodes[i], 3)
        # recieveArr = monteTransmit(EbNo, turboData, sourceCodes[i], 4)
        plt.legend()
        plt.savefig('BERSNR/' + sourceNames[i] + '/BERSNR_Comparison'+str(z)+'.png', format='png')
        plt.show()
        plt.close()
        print("")
    
        '''