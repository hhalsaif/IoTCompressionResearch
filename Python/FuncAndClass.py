#libraries
import matplotlib.pyplot as plt
import numpy as np
import sys
import zlib

# Importing the tools need from the commPy library
from commpy.utilities import hamming_dist
from commpy.channels import awgn
from commpy.channelcoding import *
from commpy.modulation import QAMModem, Modem
from sys import argv
from struct import *
# Huffman Coding in python

# Creating tree nodes
class NodeTree(object):

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return (self.left, self.right)

    def nodes(self):
        return (self.left, self.right)

    def __str__(self):
        return '%s_%s' % (self.left, self.right)

class Node:
    def __init__(self, freq,data):
        self.freq= freq
        self.data=data
        self.left = None
        self.right = None

# Main function implementing huffman coding
def huffman_code_tree(node, left=True, binString=''):
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, True, binString + '0'))
    d.update(huffman_code_tree(r, False, binString + '1'))
    return d
    
# Python program to demonstrate Huffman COmpression
def huffComp(data, z):
    print("")
    
    freq = {}
    # Calculating frequency
    for c in data:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1

    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    nodes = freq

    print(freq)
    print()

    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True) 

    f = open("huffdict/dictionary"+str(z)+".txt", 'w')
    huffmanCode = huffman_code_tree(nodes[0][0])
    print(' Char | Huffman code ')
    print('----------------------')
    for char in freq:
        print(' %-4r |%12s' % (char, huffmanCode[char]))
        f.write(' %-4r |%12s' % (char, huffmanCode[char]))
    f.close()
    
    compdata = ''
    for char in data:
        compdata += huffmanCode[char]
    return compdata, huffmanCode[len(data)]

def huffDec(data, root):
	#Enter Your Code Here
    cur = root
    chararray = []
    #For each character, 
    #If at an internal node, move left if 0, right if 1
    #If at a leaf (no children), record data and jump back to root AFTER processing character
    for char in data:
        if char == '0' and cur.left:
            cur = cur.left
        elif cur.right:
            cur = cur.right
        
        if cur.left is None and cur.right is None:
            chararray.append(cur.data)
            cur = root
    
    #Print final array
    return("".join(chararray))

# Function for LZW Compression
def LZWEnc(strData, code_width=12):
    maximum_table_size = pow(2,int(code_width))
    # Building and initializing the dictionary.
    dictionary_size = 256                   
    dictionary = {chr(i): i for i in range(dictionary_size)}    

    # We'll start off our phrase as empty and add characters to it as we encounter them
    phrase = ""         

    # This will store the sequence of codes we'll eventually write to disk 
    compressed_data = []

    # Load the text             
    data = strData
    # Iterating through the input text character by character
    for symbol in data:                     
        # Get input symbol.
        string_plus_symbol = phrase + symbol 
        
        # If we have a match, we'll skip over it
        # This is how we build up to support larger phrases
        if string_plus_symbol in dictionary: 
            phrase = string_plus_symbol
        else:
            # We'll add the existing phrase (without the breaking character) to our output
            compressed_data.append(dictionary[phrase])
            
            # We'll create a new code (if space permits)
            if(len(dictionary) <= maximum_table_size):
                dictionary[string_plus_symbol] = dictionary_size
                dictionary_size += 1
            phrase = symbol

    if phrase in dictionary:
        compressed_data.append(dictionary[phrase])

    # Storing the compressed string into a file (byte-wise).
    out = data.split(".")[0]
    output_file = open("LZW/" + out + ".lzw", "wb")
    
    encodedLZW=''
    for data in compressed_data: 
        # Saves the code as an unsigned short
        output_file.write(pack('>H',int(data)))
        encodedLZW += pack('>H',int(data))
    output_file.close()
    print(encodedLZW)
    print("")
    return encodedLZW        

def LZWDec(strData, code_width=12):
    maximum_table_size = pow(2,int(code_width))
    data = strData
    # Default values in order to read the compressed file
    compressed_data = []
    next_code = 256
    decompressed_data = ""
    phrase = ""

    # Reading the compressed file.
    while True:
        rec = file.read(2)
        if len(rec) != 2:
            break
        (data, ) = unpack('>H', rec)
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
    out = input_file_name.split(".")[0]
    output_file = open(out + "_decoded.txt", "w")
    for data in decompressed_data:
        output_file.write(data)
        decodedLZW += data
    output_file.close()
    file.close()
    return decodedLZW

def deflate(data, compresslevel=9):
    compress = zlib.compressobj(
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
    deflated = compress.compress(data)
    deflated += compress.flush()
    return deflated

def inflate(data):
    decompress = zlib.decompressobj(
            -zlib.MAX_WBITS  # see above
    )
    inflated = decompress.decompress(data)
    inflated += decompress.flush()
    return inflated

def binText(arr):
    data = ""
    for i in arr:
        data += i
    origData = ''.join(format(ord(i), 'b') for i in data)
    return origData

def hammingCoding(data):
    # Calculate the no of Redundant Bits Required
    m = len(data)
    r = calcRedundantBits(m)
    print("m = ", m)
    print("r = ", r)
    # Determine the positions of Redundant Bits
    arr = posRedundantBits(data, r)

    # Determine the parity bits
    arr = calcParityBits(arr, r)

    # Data to be transferred
    arr = np.array(list(arr), dtype=int)
    print("Data transferred is ", arr)
    print("")
    return arr

# hamming code 
def calcRedundantBits(m): 

	# Use the formula 2 ^ r >= m + r + 1 
	# to calculate the no of redundant bits. 
	# Iterate over 0 .. m and return the value 
	# that satisfies the equation 

	
	for i in range(m): 
		if(2**i >= m + i + 1):  # Use the formula 2 ^ r >= m + r + 1 
			return i 


def posRedundantBits(data, r): 

	# Redundancy bits are placed at the positions 
	# which correspond to the power of 2. 
	j = 0
	k = 1
	m = len(data) 
	res = '' 

	# If position is power of 2 then insert '0' 
	# Else append the data 
	for i in range(1, m + r+1): 
		if(i == 2**j): 
			res = res + '0'
			j += 1
		else: 
			res = res + data[-1 * k] 
			k += 1

	# The result is reversed since positions are 
	# counted backwards. (m + r+1 ... 1) 
	return res[::-1] 


def calcParityBits(arr, r): 
	n = len(arr) 

	# For finding rth parity bit, iterate over 
	# 0 to r - 1 
	for i in range(r): 
		val = 0
		for j in range(1, n + 1): 

			# If position has 1 in ith significant 
			# position then Bitwise OR the array value 
			# to find parity bit value. 
			if(j & (2**i) == (2**i)): 
				val = val ^ int(arr[-1 * j]) 
				# -1 * j is given since array is reversed 

		# String Concatenation 
		# (0 to n - 2^r) + parity bit + (n - 2^r + 1 to n) 
		arr = arr[:n-(2**i)] + str(val) + arr[n-(2**i)+1:] 
	return arr 

def detectError(arr, nr): 
    n = len(arr) 
    res = 0
  
    # Calculate parity bits again 
    for i in range(nr): 
        val = 0
        for j in range(0, n): 
            if(j & (2**i) == (2**i)): 
                val = val ^ int(arr[-1 * j]) 
  
        # Create a binary no by appending 
        # parity bits together. 
  
        res += val*(10**i) 
  
    # Convert binary to decimal 
    return int(str(res), 2)  



def correctIt(error, r, data):    
    rem = [None] * data.size
    pos_of_orisig = 0
    pos_of_redsig = 0

    if error != 0:
        if data[error-1] == 1:
            data[error -1] = 0
        else:
            data[error-1] = 1
        
        for i in range(0, data.size):
            if i==int(pow(2,pos_of_orisig)-1):
                pos_of_orisig += 1
            else:
                rem[pos_of_redsig]=data[i]
                pos_of_redsig +=1
    else:
        for i in range(0, data.size):
            if i==int(pow(2,pos_of_orisig)-1):
                pos_of_orisig += 1
            else:
                rem[pos_of_redsig]=data[i]
                pos_of_redsig +=1
    return np.array(rem)

def stringIt(arr):
    arr = arr.astype(str)
    arr = tuple(arr)
    arr = ''.join(arr)
    return arr

#Transmittion
def monteTransmit(EbNo, transArr, sourceData, code=0):
    BERarr = [None] * EbNo.size
    M = 64
    r = 0
    answer = ""
    for i in range(0, EbNo.size):
        SNR = EbNo[i]
        #reset the bit counters
        numErrs = 0
        #Simulating data transmission over a channel
        mod = QAMModem(M)

        #changing our data to a decimal for modulation
        transArr = transArr.tolist()
        transArr = int("".join(str(x) for x in transArr), 2) 

        #Changing our decimal number to a binary this is an exception. You would usually modulate a decimal/hex not a binary.
        transArr = bin(transArr).replace("0b", "")
        transArr = np.array(list(transArr), dtype=int)

        # Stimulate error in transmission by adding gaussian noise
        modArr = mod.modulate(transArr)


        """
        demodArr = bin(demodArr).replace("0b", "")
        demodArr = np.array(list(demodData), dtype=int)
        """
        
        #calculating the BER            
        
        
        if code == 1:
            answer = 'Hamming Encoded'
            r = calcRedundantBits(len(transArr))
            modArr = mod.modulate(transArr)
            rateHamming = 1 - r/(2^r - 1)
            # rateHamming = 1/2
            recieveArr = awgn(modArr, SNR, rate=rateHamming)
            demodArr = mod.demodulate(recieveArr, 'hard')
            
            r =  calcRedundantBits(len(demodArr))    
            posError = 0
            posError = detectError(stringIt(demodArr), r)
            print("The redundant bits are " + str(r))
            print("The position of error is " + str(posError))
            print("The size of our data is this " + str(demodArr.size))
            print(str(i) + " out of " + str(EbNo.size))
            # decodedData = correctIt(posError, r, demodArr)
            print("")
        
            # numErrs += np.sum(transArr != decodedData)
            # BERarr[i] = numErrs/decodedData.size
            '''
            f = open("string/hammingCodes.txt", 'w')
            f.write('Corrected Data = ' + str(demodArr))
            f.write('Original Data = ' + str(data))
            f.close()    
            '''

        elif code == 2:
            rateLDPC = len(sourceData)/len(demodArr)
            recieveArr = awgn(modArr, SNR, rate = rateLDPC)
            demodArr = mod.demodulate(recieveArr, 'hard')
            answer = 'LDPC encoded'
            decodedData = ldpc_bp_decode(demodArr, sourceData, decoder_algorithm = SPA, n_iters = 100)
            numErrs += np.sum(sourceData != decodedData)
            BERarr[i] = numErrs/decodedData.size

        elif code == 3:
            rateConvolutional = 1/2
            recieveArr = awgn(modArr, SNR, rate = rateConvolutional)
            demodArr = mod.demodulate(recieveArr, 'hard')
            answer = 'Convolutional encoded'
            decodedData = demodArr.viterbi_decode(coded_bits, trellis, tb_depth=None, decoding_type='hard')
            numErrs += np.sum(sourceData != decodedData)
            BERarr[i] = numErrs/decodedData.size

        elif code == 4:
            rateTurbo = 1/2
            recieveArr = awgn(modArr, SNR, rate = rateTurbo)
            demodArr = mod.demodulate(recieveArr, 'hard')
            answer = 'Turbo encoded'
            map_decode(sys_symbols, non_sys_symbols, trellis, noise_variance, L_int, mode='decode')
            decodedData = dturbo_decode(sys_symbols, non_sys_symbols_1, non_sys_symbols_2, trellis, noise_variance, number_iterations, interleaver, L_int=None)
            numErrs += np.sum(sourceData != decodedData)
            BERarr[i] = numErrs/decodedData.size           

        else:
            recieveArr = awgn(modArr, SNR, rate=1)
            demodArr = mod.demodulate(recieveArr, 'hard')
            answer = 'Original Data'
            numErrs += np.sum(transArr != demodArr)
            BERarr[i] = numErrs/demodArr.size
    plt.semilogy(EbNo, BERarr, label=answer)
    print("The number of errors in our code is ", numErrs)
    print("Data Transmited is ", transArr)
    print("Data Recieved is ", demodArr)
    print("The Bit error ratio is ", BERarr[i])
    print("")  
    
    return demodArr

