# libraries
import matplotlib.pyplot as plt
import numpy as np
import struct
import sys
import io
# Compression techniques
import heapq # Huffman Coding
import zlib # Deflate/Inflate
import lzma # LZMA compression
import bz2 # bzip2 compression
import zstandard # ZSTD compression
from re import sub # helper for RLE

# Importing the tools need from the commPy library
from commpy.modulation import PSKModem, Modem
from commpy.channels import awgn
from commpy.channelcoding import viterbi_decode, map_decode, turbo_decode, ldpc_bp_decode
from commpy.utilities import hamming_dist

# To help with error correction
from functools import reduce; from operator import xor 

# Global Variables
totWithHammSize = 8
noHammSize = 4

# Conversion to and from binary
def binIt(v):
    # Load data as bytes if its not otherwise continue
    binData = "".join(map(turnBin, v))
    return binData

def returnIt(b):
    return turnBack(b)

def turnBin(v):
    b = bin(v)[2:]
    return b.rjust(8, '0')

def turnBack(b):
    i = int(b,2)
    return i.to_bytes((len(b) + 7 ) // 8, byteorder = 'big')

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

    def make_entropy_dict(self, frequency):
        total = sum(frequency.values())
        entropy = {}
        for key in frequency:
            if not key in entropy:
                entropy[key] = 0
            entropy[key] += frequency[key] / total
        return entropy

    def make_heap(self, frequency):
        self.make_entropy_dict(frequency)
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
        text = inpData
        text = text.rstrip()

        frequency = self.make_frequency_dict(text)
        entropy = self.make_entropy_dict(frequency)
        self.make_heap(entropy)
        self.merge_nodes()
        self.make_codes()

        encoded_text = self.get_encoded_text(text)
        padded_encoded_text = self.pad_encoded_text(encoded_text)

        b = self.get_byte_array(padded_encoded_text)
        return b

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

def RLEEnc(inp):
    data = ''
    for i in inp:
        if i == '0': data += 'B'
        if i == '1': data += 'W'
    
    encoding = ''
    prev_char = ''
    count = 1

    if not data: return ''
    for char in data:
        # If the prev and current characters
        # don't match...
        if char != prev_char:
            # ...then add the count and character
            # to our encoding
            if prev_char:
                encoding += str(count) + prev_char
            count = 1
            prev_char = char
        else:
            # Or increment our counter
            # if the characters do match
            count += 1
    else:
        # Finish off the encoding
        encoding += str(count) + prev_char
        return encoding

def RLEDec(data):
    decode = ''
    count = ''
    for char in data:
        # If the character is numerical...
        if char.isdigit():
            # ...append it to our count
            count += char
        else:
            # Otherwise we've seen a non-numerical
            # character and need to expand it for
            # the decoding
            decode += char * int(count)
            count = ''
    
    decodedData =  ''
    for i in decode:
        if i == 'B': decodedData += '0'
        if i == 'W': decodedData += '1'
    return decodedData

h = HuffmanCoding() 
def huffComp(data):
    if type(data)!=str: str(data) 
    #data = RLEEnc(data)
    data = h.compress(data) 
    data = ''.join(map(turnBin, data))
    return data
    
def huffDecomp(data):
    data = turnBack(data)
    data = h.decompress(data)
    #data = RLEDec(data)
    print('data to be turned to bytes', data)
    data = turnBack(data)
    return data

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
        encodedLZW += struct.pack('>H', int(data))
    binData = "".join(map(turnBin, encodedLZW))
    return binData

def LZWDec(data, code_width=12): 
    dataInt = int(data,2)
    data=dataInt.to_bytes((dataInt.bit_length() + 7) // 8, byteorder='big')
    data = bytearray(data)
    maximum_table_size = pow(2,int(code_width))
    # Default values in order to read the compressed file
    compressed_data = []
    next_code = 256
    decompressed_data = ""
    phrase = ""

    dataRead=io.BytesIO(data)
    # Reading the compressed file.
    while True:
        rec = dataRead.read(2)
        if len(rec) != 2:
            break
        (data, ) = struct.unpack('>H', rec)
        compressed_data.append(data)

    # Building and initializing the dictionary.
    dictionary_size = 256
    dictionary = dict([(x, chr(x)) for x in range(dictionary_size)])

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
                dictionary[next_code] = phrase + (dictionary[code][0])
                next_code += 1
        phrase = dictionary[code]

    # storing the decompressed string into a file.
    decodedLZW = ''
    for data in decompressed_data:
        decodedLZW += data
    return decodedLZW

# FUnctions for DEFLATE Compression
def deflate(data, compresslevel=9):
    if type(data)!=bytes: data = bytes(data, 'utf-8')
    compress=zlib.compressobj(compresslevel, zlib.DEFLATED, -zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, 0 )
    data=compress.compress(data)
    data += compress.flush()
    binData = "".join(map(turnBin, data))
    return binData

def inflate(data):
    data = turnBack(data)
    decompress=zlib.decompressobj(-zlib.MAX_WBITS)
    data=decompress.decompress(data)
    data += decompress.flush()
    return data

# Functions for LZMA Compression
def LZMAComp(data):
    if type(data) != bytes: data = bytes(data,'utf-8')
    compressor = lzma.LZMACompressor()
    data = compressor.compress(data)
    binData = "".join(map(turnBin, data))
    return binData

def LZMADeComp(data):
    data = turnBack(data)
    decompressor = lzma.LZMADecompressor()
    data = decompressor.decompress(data)
    return data

# Functions for ZSTD Compression
def zstdComp(data):
    if type(data) != bytes: data = bytes(data,'utf-8')
    compressor = zstandard.ZstdCompressor()
    data = compressor.compress(data)
    binData = "".join(map(turnBin, data))
    return binData

def zstdDeComp(data):
    data = turnBack(data)   
    decompressor = zstandard.ZstdDecompressor()
    data = decompressor.decompress(data)
    return data

# Functions for bzip2
def bzipComp(data):
    if type(data) != bytes: data = bytes(data,'utf-8')
    data = bz2.compress(data)
    binData = "".join(map(turnBin, data))
    return binData

def bzipDecomp(data):
    dataInt = int(data,2)
    data= dataInt.to_bytes((dataInt.bit_length() + 7) // 8, byteorder='big')    
    data = bz2.decompress(data)
    return data

# Functions for hamming code
def hammingEnc(data):
    data = ''.join([str(i) for i in data])
    slicedStr=[]

    global noHammSize 
    global totWithHammSize

    if noHammSize == 0:
        noHammSize  = len(data)
        totWithHammSize = noHammSize + calcRedundantBits(noHammSize)

    for i in range(noHammSize, len(data)+1, noHammSize):
        slicedStr.append(data[i-noHammSize:i])
    encodedStr=[]
    for j in range(0, len(slicedStr)):
        # Calculate the no of Redundant Bits Required
        arr = [int(i) for i in slicedStr[j]]
        m=len(arr)
        r=calcRedundantBits(m)
        posRed=posRedundantBits(m, r)
        arr=calcValues(arr, posRed)
        encodedStr.append(arr)
    doneArr=''.join(encodedStr)
    # In case your data is not equal to your channel rate
    if len(data) % noHammSize != 0:
        doneArr+=data[len(data)-(len(data)%noHammSize):]
    # Data to be transferred
    doneArr=np.array(list(doneArr), dtype=int)
    return doneArr

def calcRedundantBits(m):

	# Use the formula 2 ^ r >= m + r + 1
	# to calculate the no of redundant bits.
	# Iterate over 0 .. m and return the value
	# that satisfies the equation

	for i in range(m):
		if(2**i >= m + i + 1):  # Use the formula 2 ^ r >= m + r + 1
			return i

def posRedundantBits(m, r):
    pos=[0] * r
    j = 0
    for i in range(1, m):
        if i == 2**j: 
            pos[j] = i
            j+=1
    return pos

def evenOdd(bits):
    if [i for i, bit in enumerate(bits) if bit] != []:
        return reduce(lambda x,y: x^y, [i for i, bit in enumerate(bits) if bit]) % 2 
    else:
        return 0

def calcValues(bits, posRed):   
    for i in posRed:
        val = evenOdd(bits)
        bits.insert(i, val)
    bits.insert(0, sum(bits) % 2)
    error = reduce(lambda x,y: x^y, [i for i, bit in enumerate(bits) if not bit])
    bits[error] = int(not bits[error])
    return ''.join([str(i) for i in bits])

def hammDec(recArr):
    slicedStr=[]
    extraError=False
    # breakup our data into the pieces that they were originally encoded as
    for i in range(totWithHammSize, len(recArr)+1, totWithHammSize):
        slicedStr.append(recArr[i-totWithHammSize:i])
    decodedStr=[]
    for j in range(0, len(slicedStr)):
        arr=slicedStr[j]
        arr=np.array(list(arr), dtype=int)
        # determine position of error and if none then return 0
        if [i for i, bit in enumerate(arr) if bit] != []:
            error=reduce(lambda x,y: x^y, [i for i, bit in enumerate(arr) if bit])
        else:
            error = 0
        if error == 0:
            # if no errors are detected then just remove parity bits
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
    if len(recArr) % noHammSize != 0:
        uncodedData = recArr[len(recArr)-(len(recArr)%noHammSize):]
        uncodedData = np.array(list(uncodedData), dtype=int)
        decodedStr.append(uncodedData)
    finalArr=[j for i in decodedStr for j in i]
    return finalArr #, extraError

def remParityBits(arr):
    # remove parity bits
    k=0
    decodedArr = []
    for i in range(1, len(arr)):
        if i == 2**k: k += 1
        else: decodedArr.append(arr[i])
    decodedArr=decodedArr[1:]
    return arr

# Transmittion
deCompAlgo = [returnIt, huffDecomp, inflate, LZMADeComp, zstdDeComp, bzipDecomp] # Functions for decompression techniques we use
def monteTransmit(EbNo, transArr, sourceData, code=0, source=0):
    BERarr=[None] * len(EbNo)
    M=64
    numErrs=0
    answer=""
    sourceData = ''.join(map(turnBin, sourceData))
    sourceData = np.array(list(sourceData), dtype=int)
    for i in range(0, len(EbNo)):
        SNR=EbNo[i]
        # Simulating data transmission over a channel
        mod=PSKModem(M)
        modArr=mod.modulate(transArr)
        # recieving and decoding data
        if code == 1:
            answer='Hamming Encoded'
            r = totWithHammSize - noHammSize
            rateHamming= 1-(r/(2**r -1))
            recieveArr=awgn(modArr, SNR, rate=rateHamming)
            demodArr=mod.demodulate(recieveArr, 'hard')
            decodedData=hammDec(demodArr)
            decodedData=''.join([str(i) for i in decodedData])

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

        if code == 0:
            answer='Original Data'
            recieveArr=awgn(modArr, SNR, rate=1)
            demodArr=mod.demodulate(recieveArr, 'hard')
            decodedData=''.join([str(i) for i in demodArr])
        # decodedData = deCompAlgo[source](decodedData) #Decompress our to the original source
        # decodedData = ''.join(map(turnBin, decodedData))
        decodedData = np.array(list(decodedData), dtype=int)
        numErrs += np.sum(decodedData != transArr)
        BERarr[i]=numErrs/len(transArr)
    plt.semilogy(EbNo[::-1], BERarr, label=answer)
    print(answer)
    print("The number of errors in our code is ", numErrs)
    print("The Bit error ratio is ", BERarr[i])
    print("\n")
    return demodArr
