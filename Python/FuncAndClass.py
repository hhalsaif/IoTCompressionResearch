# libraries
import matplotlib.pyplot as plt
import numpy as np
import heapq
import zlib
import sys
import io

# Importing the tools need from the commPy library
from commpy.modulation import QAMModem, Modem
from commpy.channels import awgn
from commpy.channelcoding import viterbi_decode, map_decode, turbo_decode, ldpc_bp_decode
from functools import reduce; from operator import ixor
from struct import *

# Huffman Coding in python

# Global Variables
totWithHammSize = 8
noHammSize = 4

# Conversion to and from binary


def binText(arr):
     # Load data as bytes if its not otherwise continue
    if type(arr) != bytes: arr = bytes(arr, 'utf-8')
    return bin(int.from_bytes(arr, byteorder=sys.byteorder))[2:]


def textBin(arr):
    arr = int(arr, 2).to_bytes((len(arr) + 7) // 8, byteorder=sys.byteorder)
    return arr.decode()


# Huffman Coding
class HuffmanCoding:
    '''
    A lot of the code credit goes to
    author: Bhrigu Srivastava
    website: https:bhrigu.me
    '''

    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        # defining comparators less_than and equals
        def __lt__(self, other):
            return self.freq < other.freq

        def __eq__(self, other):
            if(other == None):
                return False
            if(not isinstance(other, HeapNode)):
                return False
            return self.freq == other.freq

    # functions for compression

    def make_frequency_dict(self, text):
        frequency = {}
        for character in text:
            if not character in frequency:
                frequency[character] = 0
            frequency[character] += 1
        return frequency

    def make_heap(self, frequency):
        for key in frequency:
            node = self.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while(len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if(root == None):
            return

        if(root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self, text):
        encoded_text = ""
        for character in text:
            encoded_text += self.codes[character]
        return encoded_text

    def pad_encoded_text(self, encoded_text):
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text
        return encoded_text

    def get_byte_array(self, padded_encoded_text):
        if(len(padded_encoded_text) % 8 != 0):
            print("Encoded text not padded properly")
            exit(0)

        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i+8]
            b.append(int(byte, 2))
        return b

    def compress(self, inpData):
        inpData = str(inpData, 'utf-8')
        text = inpData
        text = text.rstrip()

        frequency = self.make_frequency_dict(text)
        self.make_heap(frequency)
        self.merge_nodes()
        self.make_codes()

        encoded_text = self.get_encoded_text(text)
        padded_encoded_text = self.pad_encoded_text(encoded_text)

        b = self.get_byte_array(padded_encoded_text)
        return bin(int.from_bytes(bytes(b), byteorder=sys.byteorder))[2:]

    # functions for decompression

    def remove_padding(self, padded_encoded_text):
        padded_info=padded_encoded_text[:8]
        extra_padding=int(padded_info, 2)
        padded_encoded_text=padded_encoded_text[8:]
        encoded_text=padded_encoded_text[:-1*extra_padding]
        return encoded_text

    def decode_text(self, encoded_text):
        current_code=""
        decoded_text=""
        for bit in encoded_text:
            current_code += bit
            if(current_code in self.reverse_mapping):
                character=self.reverse_mapping[current_code]
                decoded_text += character
                current_code=""
        return decoded_text

    def decompress(self, data):
        bit_string=""
        data = int(data, 2).to_bytes((len(data) + 7) // 8, byteorder=sys.byteorder)
        byteData=io.BytesIO(data)
        byte=byteData.read(1)
        while(len(byte) > 0):
            byte=ord(byte)
            bits=bin(byte)[2:].rjust(8, '0')
            bit_string += bits
            byte=byteData.read(1)
        encoded_text=self.remove_padding(bit_string)
        decompressed_text=self.decode_text(encoded_text)
        return decompressed_text

# Function for LZW Compression
'''
A lot of code credit goes to:
Author = Aryaman Sharda
website = https://hackernoon.com/unixs-lzw-compression-algorithm-how-does-it-work-cp65347h
'''
def LZWEnc(data, code_width=12):
    maximum_table_size=pow(2, int(code_width))
    # Building and initializing the dictionary.
    dictionary_size=256
    dictionary={chr(i): i for i in range(dictionary_size)}

    # Load data as text if its not text otherwise continue
    if type(data) != str: data=str(data, 'utf-8')

    # We'll start off our phrase as empty and add characters to it as we encounter them
    phrase=""

    # This will store the sequence of codes we'll eventually write to disk
    compressed_data=[]

    # Iterating through the input text character by character
    for symbol in data:
        # Get input symbol.
        string_plus_symbol=phrase + symbol

        # If we have a match, we'll skip over it
        # This is how we build up to support larger phrases
        if string_plus_symbol in dictionary:
            phrase=string_plus_symbol
        else:
            # We'll add the existing phrase (without the breaking character) to our output
            compressed_data.append(dictionary[phrase])

            # We'll create a new code (if space permits)
            if(len(dictionary) <= maximum_table_size):
                dictionary[string_plus_symbol]=dictionary_size
                dictionary_size += 1
            phrase=symbol

    if phrase in dictionary:
        compressed_data.append(dictionary[phrase])

    # Storing the compressed string into a file (byte-wise).
    encodedLZW=bytearray()
    for data in compressed_data:
        # Saves the code as an unsigned short
        encodedLZW += pack('>H', int(data))
    return bin(int.from_bytes(encodedLZW, byteorder=sys.byteorder))[2:]

def LZWDec(data, code_width=12):
    data=int(data, 2).to_bytes((len(data) + 7) // 8, byteorder=sys.byteorder)
    maximum_table_size=pow(2, int(code_width))

    # Default values in order to read the compressed file
    compressed_data=[]
    next_code=256
    decompressed_data=""
    phrase=""

    dataInput=io.BytesIO(data)
    while True:
        rec=dataInput.read(2)
        if len(rec) != 2:
            break
        (data, )=unpack('>H', rec)
        compressed_data.append(data)

    # Building and initializing the dictionary.
    dictionary_size=256
    dictionary=dict([(x, chr(x)) for x in range(dictionary_size)])

    # Iterating through the codes.
    # LZW Decompression algorithm
    for code in compressed_data:
        # If we find a new
        if not (code in dictionary):
            dictionary[code] = phrase + (phrase[0])

        decompressed_data += dictionary[code]

        if not(len(phrase) == 0):
            # Ensures we don't exceed the bounds of the table
            if(len(dictionary) <= maximum_table_size):
                dictionary[next_code]=phrase + (dictionary[code][0])
                next_code += 1
        phrase=dictionary[code]

    # storing the decompressed string into a file.
    decodedLZW=''
    for data in decompressed_data:
        decodedLZW += data
    return decodedLZW

'''
A lot of code credit goes to:
Author = James Cameron
website = https://izziswift.com/python-inflate-and-deflate-implementations/
'''

def deflate(data, compresslevel=9):
    compress=zlib.compressobj(
            compresslevel,        # level: 0-9
            zlib.DEFLATED,        # method: must be DEFLATED
            -zlib.MAX_WBITS,      # window size in bits:
                                  #   -15..-8: negate, suppress header
                                  #   8..15: normal
                                  #   16..30: subtract 16, gzip header
            zlib.DEF_MEM_LEVEL,   # mem level: 1..8/9
            0                     # strategy:
                                  #   0 = Z_DEFAULT_STRATEGY
                                  #   1 = Z_FILTERED
                                  #   2 = Z_HUFFMAN_ONLY
                                  #   3 = Z_RLE
                                  #   4 = Z_FIXED
    )
    deflated=compress.compress(data)
    deflated += compress.flush()
    return bin(int.from_bytes(deflated, byteorder=sys.byteorder))[2:]

def inflate(data):
    decompress=zlib.decompressobj(
            -zlib.MAX_WBITS  # see above
    )
    data=int(data, 2).to_bytes((len(data) + 7) // 8, byteorder=sys.byteorder)
    inflated=decompress.decompress(data)
    inflated += decompress.flush()
    return str(inflated, 'utf-8')

def hammingCoding(data):
    slicedStr=[]
    for i in range(noHammSize, len(data)+1, noHammSize):
        slicedStr.append(data[i-noHammSize:i])
    encodedStr=[]
    for j in range(0, len(slicedStr)):
        # Calculate the no of Redundant Bits Required
        m=len(slicedStr[j])
        r=calcRedundantBits(m)
        arr=addParityBits(slicedStr[j], r)
        arr=parityValues(arr)
        encodedStr.append(arr)
    doneArr=''.join(encodedStr)
    # Data to be transferred
    doneArr=np.array(list(doneArr), dtype=int)
    print("")
    return doneArr

def calcRedundantBits(m):

	# Use the formula 2 ^ r >= m + r + 1
	# to calculate the no of redundant bits.
	# Iterate over 0 .. m and return the value
	# that satisfies the equation


	for i in range(m):
		if(2**i >= m + i + 1):  # Use the formula 2 ^ r >= m + r + 1
			return i

def addParityBits(data, r):
    m=len(data)
    data=list(data)
    # if in postition that is a power of 2 then insert 0
    data.insert(0, '0')
    j=0
    for i in range(1, m+1):
        if i == 2**j:
            data.insert(i, '0')
            j += 1
    data=''.join(data)
    return data

def parityValues(arr):
    arr=[int(i) for i in list(arr)]
    parityNo=reduce(ixor, arr)
    parityNo=bin(parityNo).replace("0b", "")
    parityNo=[int(i) for i in list(parityNo)]
    for i in range(0, len(parityNo)): arr[2**i]=parityNo[i]
    total=sum(arr)
    if total % 2 == 0: arr[0]=0
    else: arr[0]=1
    arr=''.join([str(i) for i in arr])
    return arr

def hammDec(recArr):
    slicedStr=[]
    extraError=False
    # breakup our data into the pieces that they were originally encoded as
    for i in range(totWithHammSize, len(recArr)+1, totWithHammSize):
        slicedStr.append(recArr[i-totWithHammSize:i])
    decodedStr=[]
    for j in range(0, len(slicedStr)):
        arr=slicedStr[j]
        arr=[int(i) for i in list(arr)]
        # determine position of error and if none then return 0
        error=reduce(ixor, arr)
        if error == 0:
            # if this is true then no errors detected then just remove parity bits
            decodedArr=remParityBits(arr)
            decodedStr.append(decodedArr)
        else:
            # correct error
            arr[error]=int(not arr[error])
            # detect for additional error
            total=sum(arr)
            if total % 2 != arr[0]: extraError=True
            # remove any parity bits
            decodedArr=remParityBits(arr)
            decodedStr.append(decodedArr)
    finalArr=[j for i in decodedStr for j in i]
    return finalArr, extraError

def remParityBits(arr):
    # remove parity bits
    k=0
    decodedArr=[]
    for i in range(1, len(arr)):
        if i != 2**k:
            decodedArr.append(arr[i])
            k += 1
    decodedArr=decodedArr[1:]
    return decodedArr

# Transmittion
def monteTransmit(EbNo, transArr, sourceData, code=0, source=0):
    BERarr=[None] * len(EbNo)
    M=64
    numErrs=0
    answer=""
    for i in range(0, len(EbNo)):
        SNR=EbNo[i]
        # Simulating data transmission over a channel
        mod=QAMModem(M)

        # Stimulate error in transmission by adding gaussian noise
        modArr=mod.modulate(transArr)

        if code == 1:
            answer='Hamming Encoded'
            modArr=mod.modulate(transArr)
            rateHamming=noHammSize/totWithHammSize
            recieveArr=awgn(modArr, SNR, rate=rateHamming)
            demodArr=mod.demodulate(recieveArr, 'hard')
            decodedData, extra=hammDec(demodArr)
            decodedData=''.join(str(i) for i in decodedData)

        elif code == 2:
            rateLDPC=len(sourceData)/len(demodArr)
            recieveArr=awgn(modArr, SNR, rate=rateLDPC)
            demodArr=mod.demodulate(recieveArr, 'hard')
            answer='LDPC encoded'
            decodedData=ldpc_bp_decode(
                demodArr, sourceData, decoder_algorithm=SPA, n_iters=100)
            numErrs += np.sum(sourceData != decodedData)
            BERarr[i]=numErrs/decodedData.size

        elif code == 3:
            rateConvolutional=1/2
            recieveArr=awgn(modArr, SNR, rate=rateConvolutional)
            demodArr=mod.demodulate(recieveArr, 'hard')
            answer='Convolutional encoded'
            decodedData=demodArr.viterbi_decode(
                coded_bits, trellis, tb_depth=None, decoding_type='hard')
            numErrs += np.sum(sourceData != decodedData)
            BERarr[i]=numErrs/decodedData.size

        elif code == 4:
            rateTurbo=1/2
            recieveArr=awgn(modArr, SNR, rate=rateTurbo)
            demodArr=mod.demodulate(recieveArr, 'hard')
            answer='Turbo encoded'
            map_decode(sys_symbols, non_sys_symbols, trellis,
                       noise_variance, L_int, mode='decode')
            decodedData=dturbo_decode(sys_symbols, non_sys_symbols_1, non_sys_symbols_2,
                                      trellis, noise_variance, number_iterations, interleaver, L_int=None)
            numErrs += np.sum(sourceData != decodedData)
            BERarr[i]=numErrs/decodedData.size

        else:
            answer='Original Data'
            recieveArr=awgn(modArr, SNR, rate=1)
            demodArr=mod.demodulate(recieveArr, 'hard')
            decodedData=''.join(str(i) for i in demodArr)

        if source == 1:
            h=HuffmanCoding()
            decodedData=h.decompress(decodedData)
        elif source == 2: decodedData=LZWDec(decodedData)
        elif source == 3: decodedData=inflate(decodedData)
        # else: decodedData=textBin(decodedData)

        numErrs += np.sum(binText(decodedData) != binText(sourceData))
        BERarr[i]=numErrs/len(binText(decodedData))
    plt.semilogy(EbNo[::-1], BERarr, label=answer)
    print(answer)
    print("The number of errors in our code is ", numErrs)
    # print("Data Transmited is", sourceData)
    # print("Data Recieved is", decodedData)
    print("The Bit error ratio is ", BERarr[i])
    print("")

    return demodArr
