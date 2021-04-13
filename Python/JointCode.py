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
    sizeOfData = 4**2 #np.random.randint(1, 64)
    symbols = list(string.ascii_uppercase)
    arr = np.random.choice(symbols, sizeOfData) # The code
    strData = ''.join([str(i) for i in arr])
    
    # Data generation of random number from the dataset
    df = pd.read_csv("Datasets/IOT-temp.csv")    
    temps = pd.Series(df['temp'])
    dataNo = np.random.randint(1,97600)
    data = temps[dataNo]
    byteData = data.tobytes()
    # byteData = bytes(data, 'utf-8')

    # How many compression techniques do we want to use 
    noOfSources = 6
    # How many Channel codes do we want to use
    noOfChannels = 1

    # Data that will help with comparisions of compression techniques
    sourceNames = ["Original", "Huffman","DEFLATE", "LZMA", "Zstandard", "bz2"]
    compAlgo = [binIt, huffComp, deflate, LZMAComp, zstdComp, bzipComp] # Functions for the compression techniques we use
    deCompAlgo = [returnIt, huffDecomp, inflate, LZMADeComp, zstdDeComp, bzipDecomp] # Functions for decompression techniques we use
    sourceCodes=[0] * noOfSources 
    decodedCodes=[0] * noOfSources

    # data that will help with comparision of channel codes
    channelNames = ["Original", "Hamming"]
    channelAlgo = [hammingEnc] # Functions for the channel codes we use
    deChannelAlgo = [hammDec] # Functions for the channel codes we use
    channelCodes=[0] * noOfChannels 
    decodedChannels=[0] * noOfChannels

    # Tracking for benchmarking
    timeEn = [0.0] * noOfSources ; timeDe = [0.0] * noOfSources ; flopsEn = [0.0] * noOfSources; flopsDe = [0.0] * noOfSources

    # Benchmarking encoding of compression
    for i in range(noOfSources):
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
    # Benchmarking decoding of compression
    for i in range(noOfSources):     
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
    for i in range(0, noOfSources):
        print("Using ", sourceNames[i], " The size is ", str(len(sourceCodes[i])) + ":")
        print("Compression ratio of " , len(sourceCodes[0])/len(sourceCodes[i]))
        print("The time taken to compress", timeEn[i])
        print("The time taken to decompress", timeDe[i])
        print("Number of FLOPS to compress", flopsEn[i])
        print("Number of FLOPS to decompress", flopsDe[i])
        print("\n")
        # plotting for better visuals
        plt.bar(sourceNames[i], len(sourceCodes[i]), align='center')
        plt.title('Data size comparison')
        plt.savefig('CompSize/Source_Comparison'+str(z)+'.png', format='png')
    plt.show()
    plt.close()

    for i in range(0,noOfSources): sourceCodes[i] = np.array(list(sourceCodes[i]),dtype=int)



    sourceData = sourceCodes[1]
    # Benchmarking encoding of channel codes
    for i in range(noOfChannels):
        # Time taken to compress
        timeStart = perf_counter_ns()
        channelCodes[i] = channelAlgo[i](sourceData)
        timeStop = perf_counter_ns()
        timeEn[i] = timeStop - timeStart

        '''
        #Number of Flops 
        papi_high.Flops() 
        result = papi_high.flops(rtime=0,ptime=0,flpops=0,mflops=0)
        flopsEn[i] = result.mflops
        papi_high.stop_counters()
        '''        

    # Benchmarking decoding of channel codes
    for i in range(noOfChannels):
        # Add error cuz
        error = np.random.randint(1, len(channelCodes[i]))
        channelCodes[i][error] = int(not channelCodes[i][error])
        
        # Time taken to compress
        timeStart = perf_counter_ns()
        decodedChannels[i] = deChannelAlgo[i](channelCodes[i])
        timeStop = perf_counter_ns()
        timeEn[i] = timeStop - timeStart
        
        '''
        #Number of Flops 
        papi_high.Flops() 
        result = papi_high.flops(rtime=0,ptime=0,flpops=0,mflops=0)
        flopsEn[i] = result.mflops
        papi_high.stop_counters()
        '''        

    print("")
    for i in range(noOfChannels):    
        if sorted(decodedChannels[i]) == sorted(sourceData):
            print("decoded properly")
        print("Using ", channelNames[i+1], " The size is ", str(len(channelCodes[i])) + ":")
        print("Channel rate" , len(channelCodes[i])/len(sourceData))
        print("The time taken to encode", timeEn[i])
        print("The time taken to decoded", timeDe[i])
        print("Number of FLOPS to encode", flopsEn[i])
        print("Number of FLOPS to decode", flopsDe[i])
        print("\n")



    '''
    # Transmission
    for i in range(1, noOfSources):
        # hamming code
        origData = sourceCodes[0]
        hammData = hammingEnc(sourceCodes[i])
        # LDPCData = get_ldpc_code_params(sourceCodes[i])
        # classTrellis(memory, g_matrix, feedback=0, code_type='default')
        # convData = conv_encode(sourceCodes[i], trellis, termination='term', puncture_matrix=None)
        # turboData = turbo_encode(sourceCodes[i], trellis1, trellis2, interleaver)

        EbNo = np.arange(60, 63)
        plt.xlabel('EbNo(dB)')
        plt.ylabel('BER')
        plt.title('BER vs SNR')
        plt.yscale('log')
        plt.grid(True)

        print(sourceNames[i]+ ':')
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
    '''