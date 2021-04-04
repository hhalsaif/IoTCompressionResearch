# Importing general libraries 
import matplotlib.pyplot as plt
import numpy as np
from commpy.channelcoding import Trellis, conv_encode, turbo_encode, get_ldpc_code_params
import string
from time import perf_counter
# Importing premade functions that will help clean the code
from FuncAndClass import *

for z in range(10):
    print(z)
    # Data Generation
    sizeOfData = 20  #np.random.randint(1, 64)
    symbols = list(string.ascii_uppercase)
    arr = np.random.choice(symbols, sizeOfData) # The code
    strData = ''.join([str(i) for i in arr])
    byteData = bytes(strData, 'utf-8')

    # Data that will help with comparisions
    sourceNames = ["Original Data", "Huffman Compression", "LZW Compression", "Inflate/Deflate Compression"]
    origData = binText(strData) ; h = HuffmanCoding() ; huffData = h.compress(byteData) ; LZWdata = LZWEnc(byteData) ; defData = deflate(byteData)
    sourceCodes=[origData, huffData, LZWdata, defData] 
    decodedCodes=[origData, huffData, LZWdata, defData]
    compAlgo = [binText, h.compress, LZWEnc, deflate] # Functinos of the compression techniques we use
    deCompAlgo = [textBin, h.decompress, LZWDec, inflate] # Functions of decompression techniques we use
    noOfData = 4
    time =  [0.0, 0.0, 0.0, 0.0] ; timeDe = [0.0, 0.0, 0.0, 0.0] ;timeStart = [0.0, 0.0, 0.0, 0.0] ; timeStop = [0.0, 0.0, 0.0, 0.0]

    for i in range(noOfData):
        timeStart[i] = perf_counter()
        sourceCodes[i] = compAlgo[i](byteData)
        timeStop[i] = perf_counter()
        time[i] = timeStop[i] - timeStart[i]
    
    for i in range(noOfData):
        timeStart[i] = perf_counter()
        decodedCodes[i] = deCompAlgo[i](sourceCodes[i])
        timeStop[i] = perf_counter()
        timeDe[i] = timeStop[i] - timeStart[i]
    
    for i in range(noOfData): 
        if strData != decodedCodes[i]: print(str(i) + "\n" + "Error in decoding found" + "\n" + "original data: " + strData + "\n" + "decoded data: " + decodedCodes[i])
    
    print ("Normally our code would be of size ", len(sourceCodes[0])) ; print("")
    for i in range(0, noOfData):
        print ("Using ", sourceNames[i],  len(sourceCodes[i]))
        print ("Compression ratio of " , len(origData)/len(sourceCodes[i]))
        print ("The time taken to compress", time[i])
        print ("The time taken to decompress", timeDe[i])
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
    

# In[ ]:




