#!/usr/bin/env python
# coding: utf-8

# Importing general libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string

# Transmission simulation
from commpy.channelcoding import Trellis, conv_encode, turbo_encode, get_ldpc_code_params # encoders
from commpy.channelcoding import viterbi_decode, map_decode, turbo_decode, ldpc_bp_decode # decoders
from commpy.modulation import PSKModem, Modem
from commpy.channels import awgn
from commpy.utilities import hamming_dist, signal_power
# Importing premade functions that will help clean the code
from FuncAndClass import *
# Importing tools for benchmarking
from time import perf_counter_ns
from pypapi import papi_high

for z in range(10):
    print(z)
    # Data Generation of random string
    sizeOfData = 4**4 #np.random.randint(1, 64)
    symbols = list(string.ascii_uppercase)
    arr = np.random.choice(symbols, sizeOfData) # The code
    strData = ''.join([str(i) for i in arr])
    
    # Data generation of random number from the dataset
    df = pd.read_csv("Datasets/IOT-temp.csv")    
    temps = pd.Series(df['temp'])
    dataNo = np.random.randint(1,97600)
    data = temps[dataNo]
    #data = strData
    byteData = data.tobytes()
    #byteData = bytes(data, 'utf-8')

    # How many compression techniques do we want to use 
    noOfSources = 6
    # How many Channel codes do we want to use
    noOfChannels = 2

    # Data that will help with comparisions of compression techniques
    sourceNames = ["Original", "Huffman","DEFLATE", "LZMA", "Zstandard", "bz2"]
    compAlgo = [binIt, huffComp, deflate, LZMAComp, zstdComp, bzipComp] # Functions for the compression techniques we use
    deCompAlgo = [returnIt, huffDecomp, inflate, LZMADeComp, zstdDeComp, bzipDecomp] # Functions for decompression techniques we use
    sourceCodes=[0] * noOfSources 
    decodedCodes=[0] * noOfSources
    sigPower = [0] * noOfSources
    # data that will help with comparision of channel codes
    channelNames = ["Original", "Hamming"]
    channelAlgo = [hammingEnc] # Functions for the channel codes we use
    deChannelAlgo = [hammDec] # Functions for the channel codes we use
    channelCodes=[0] * noOfChannels 
    decodedChannels=[0] * noOfChannels
    BER=[0] * noOfChannels
    rate = [0] * noOfChannels
    # Tracking for benchmarking
    timeEn = [0.0] * noOfSources ; timeDe = [0.0] * noOfSources ; flopsEn = [0.0] * noOfSources; flopsDe = [0.0] * noOfSources

    transData =  ''.join(map(turnBin, byteData))

    # Benchmarking encoding of compression
    for i in range(noOfSources):
        # Time taken to compress
        timeStart = perf_counter_ns()
        if i == 1: sourceCodes[i] = compAlgo[i](transData)
        else : sourceCodes[i] = compAlgo[i](byteData)
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

    for i in range (0, noOfSources):
        M = 16
        mod=PSKModem(M)
        sigPower[i]=signal_power(np.array(list(sourceCodes[i]), dtype=int))
    
    print("Normally our code would be of size ", len(sourceCodes[0])) ; print("")
    for i in range(0, noOfSources):
        if decodedCodes[i] != decodedCodes[0]:
            print('There is an error in', sourceNames[i])
            print('Original Code is', decodedCodes[0])
            print('Recieved Code is', decodedCodes[i])
            print("")
        print("Using ", sourceNames[i], " The size is ", str(len(sourceCodes[i])) + ":")
        print("Compression ratio of " , len(sourceCodes[0])/len(sourceCodes[i]))
        print("The time taken to compress", timeEn[i])
        print("The time taken to decompress", timeDe[i])
        print("Number of FLOPS to compress", flopsEn[i])
        print("Number of FLOPS to decompress", flopsDe[i])
        print("Estimated Saved Energy", (sigPower[i] - sigPower[0]))
        print("\n")
        # plotting for better visuals
        plt.bar(sourceNames[i], len(sourceCodes[i]), align='center')
        plt.title('Data size comparison')
        plt.savefig('CompSize/Source_Comparison'+str(z)+'.png', transparent=True, format='png')
    plt.show()
    plt.close()
    
    '''
    sourceData = sourceCodes[1]
    # Benchmarking encoding of channel codes
    for i in range(noOfChannels):
        # Time taken to compress
        timeStart = perf_counter_ns()
        channelCodes[i] = channelAlgo[i](sourceData)
        timeStop = perf_counter_ns()
        timeEn[i] = timeStop - timeStart

        
        #Number of Flops 
        #papi_high.Flops() 
        #result = papi_high.flops(rtime=0,ptime=0,flpops=0,mflops=0)
        #flopsEn[i] = result.mflops
        #papi_high.stop_counters()      
        
        rate[i] = 1/2

        M = 64
        mod=PSKModem(M)
        channelCodes[i]=mod.modulate(channelCodes[i])
        
    # Benchmarking decoding of channel codes
    for i in range(noOfChannels):
        # Add error cuz
        SNR = 10   
        print(len(channelCodes[i]))
        print(len(sourceData))
        channelCodes[i] = awgn(channelCodes[i], SNR, rate=rate[i])
        channelCodes[i] = mod.demodulate(channelCodes[i], 'hard')

        # Time taken to compress
        timeStart = perf_counter_ns()
        decodedChannels[i] = deChannelAlgo[i](channelCodes[i])
        timeStop = perf_counter_ns()
        timeEn[i] = timeStop - timeStart
        
        #Number of Flops 
        #papi_high.Flops() 
        #result = papi_high.flops(rtime=0,ptime=0,flpops=0,mflops=0)
        #flopsEn[i] = result.mflops
        #papi_high.stop_counters()
        
        print(decodedChannels[i])
        print(sourceData)
        numErrs = hamming_dist(decodedChannels[i], sourceData)
        BER[i]=numErrs/len(sourceData)
        
    print("")
    for i in range(noOfChannels):    
        print("Using ", channelNames[i+1], " The size is ", str(len(channelCodes[i])) + ":")
        print("Channel rate" , len(channelCodes[i])/len(sourceData))
        print("Bit Error Rate", BER[i])
        print("The time taken to encode", timeEn[i])
        print("The time taken to decoded", timeDe[i])
        print("Number of FLOPS to encode", flopsEn[i])
        print("Number of FLOPS to decode", flopsDe[i])
        print("\n")
    
    '''
    #Transmission
    i=1
    sourceData = np.array(list(sourceCodes[i]), dtype=int)
    j=1
    origData = sourceData
    protectedData = channelAlgo[j-1](sourceData)

    EbNo = np.arange(-40, 20)
    plt.xlabel('EbNo(dB)')
    plt.ylabel('BER')
    plt.title('BER vs SNR')
    plt.yscale('log')
    plt.grid(True)

    print(sourceNames[i]+ ':')
    monteTransmit(EbNo, origData, decodedCodes[0], code=0, source=1)
    recieveArr = monteTransmit(EbNo, protectedData, decodedCodes[i], code=j, source=1)

    plt.legend()
    plt.savefig('BERSNR/' + sourceNames[i] + '/BERSNR_Comparison'+str(z)+ channelNames[j] + '.png', transparent=True, format='png')
    plt.show()
    plt.close()
    print("")
    

