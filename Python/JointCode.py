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
from pypapi import events as papi_events

for z in range(10):
    print(z)
    # Data Generation
    sizeOfData = 40  #np.random.randint(1, 64)
    symbols = list(string.ascii_uppercase)
    arr = np.random.choice(symbols, sizeOfData) # The code
    strData = ''.join([str(i) for i in arr])
    byteData = bytes(strData, 'utf-8')

    
    # Data that will help with comparisions of compression techniques
    sourceNames = ["Original Data", "Huffman Compression", "LZW Compression", "Inflate/Deflate Compression"]
    origData = binText(strData) ; h = HuffmanCoding() ; huffData = h.compress(byteData) ; LZWdata = LZWEnc(byteData) ; defData = deflate(byteData)
    sourceCodes=[origData, huffData, LZWdata, defData] 
    decodedCodes=[origData, huffData, LZWdata, defData]
    compAlgo = [binText, h.compress, LZWEnc, deflate] # Functinos of the compression techniques we use
    deCompAlgo = [textBin, h.decompress, LZWDec, inflate] # Functions of decompression techniques we use
    
    # Tracking for benchmarking
    timeEn = [0.0,0.0,0.0,0.0] ; timeDe = [0.0,0.0,0.0,0.0]; flopsEn = [0.0,0.0,0.0,0.0,]; flopsDe = [0.0,0.0,0.0,0.0]
    
    noOfData = 4    #How many compression techniques do we want to use 


    for i in range(noOfData):
        # Time taken to compress
        timeStart = perf_counter_ns()
        if i == 0: sourceCodes[i] = compAlgo[i](strData)
        else:sourceCodes[i] = compAlgo[i](byteData)
        timeStop = perf_counter_ns()
        timeEn[i] = timeStart - timeStop
        
        #Number of Flops 
        papi_high.flops()
        result = papi_high.flops()
        flopsEn[i] = result.mflops
        papi_high.stop_counters()

    for i in range(noOfData):     
        # Time taken to decompress
        timeStart = perf_counter_ns()
        decodedCodes[i] = deCompAlgo[i](sourceCodes[i])
        timeStop = perf_counter_ns()
        timeDe[i] = timeStart - timeStop
        
        #Number of Flops 
        papi_high.flops()
        decodedCodes[i] = deCompAlgo[i](sourceCodes[i])
        result = papi_high.flops()
        flopsDe[i] = result.mflops
        papi_high.stop_counters()
    
    for i in range(noOfData): 
        if strData != decodedCodes[i]: print(str(i) + "\n" + "Error in decoding found" + "\n" + "original data: " + strData + "\n" + "decoded data: " + decodedCodes[i])
    
    print("Normally our code would be of size ", len(sourceCodes[0])) ; print("")
    for i in range(0, noOfData):
        print("Using ", sourceNames[i],  len(sourceCodes[i]))
        print("Compression ratio of " , len(origData)/len(sourceCodes[i]))
        print("The time taken to compress", timeEn[i])
        print("The time taken to decompress", timeDe[i])
        print("Number of FLOPS to compress", flopsEn[i])
        print("Number of FLOPS to compress", flopsDe[i])
        print("")
        # plotting for better visuals
        plt.bar(sourceNames[i], len(sourceCodes[i]), align='center')
        plt.title('Data size comparison')
        plt.savefig('CompSize/Source_Comparison'+str(z)+'.png', format='png')
    plt.show()
    plt.close()

    for i in range(1, noOfData):
        # hamming code
        origData = np.array(list(sourceCodes[0]),dtype=int)
        hammData = hammingCoding(sourceCodes[i])
        #LDPCData = get_ldpc_code_params(sourceCodes[i])
        # classTrellis(memory, g_matrix, feedback=0, code_type='default')
        # convData = conv_encode(sourceCodes[i], trellis, termination='term', puncture_matrix=None)
        # turboData = turbo_encode(sourceCodes[i], trellis1, trellis2, interleaver)

        EbNo = np.arange(-10, 20)
        plt.xlabel('EbNo(dB)')
        plt.ylabel('BER')
        plt.title('BER vs SNR')
        plt.yscale('log')
        plt.grid(True)

        monteTransmit(EbNo, origData, strData)
        recieveArr = monteTransmit(EbNo, hammData, strData, 1, i)
        # recieveArr = monteTransmit(EbNo, dict(subString.split("=") for subString in sourceCodes[i].split(";")) + LDPCData, LDPCData, 2)
        # recieveArr = monteTransmit(EbNo, convData, sourceCodes[i], 3)
        # recieveArr = monteTransmit(EbNo, turboData, sourceCodes[i], 4)
        plt.legend()
        plt.savefig('BERSNR/' +sourceNames[i] + '/BERSNR_Comparison'+str(z)+'.png', format='png')
        plt.show()
        plt.close()
        print("")
    



