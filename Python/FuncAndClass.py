#libraries
import matplotlib.pyplot as plt
import numpy as np
# Importing the tools need from the commPy library
from commpy.utilities import hamming_dist
from commpy.channels import awgn
from commpy.modulation import QAMModem, Modem

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


# Main function implementing huffman coding
def huffman_code_tree(node, left=True, binString=''):
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, True, binString + '0'))
    d.update(huffman_code_tree(r, False, binString + '1'))
    return d
# Python program to dmeonstrate 

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
def monteTransmit(EbNo, transArr, data=[0, 1, 2]):
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
        

        """
        demodArr = bin(demodArr).replace("0b", "")
        demodArr = np.array(list(demodData), dtype=int)
        """
        
        #calculating the BER            
        
        
        if data!=[0, 1, 2]:
            r = calcRedundantBits(len(transArr))
            modArr = mod.modulate(transArr)
            # rateHamming = 1 - r/(2^r - 1)
            rateHamming = 1/2
            recieveArr = awgn(modArr, SNR, rate=rateHamming)
            demodArr = mod.demodulate(recieveArr, 'hard')

            answer = 'Encoded Data'
            data = transArr
            decodedData = demodArr
            '''
            r =  calcRedundantBits(len(demodArr))    
            posError = 0
            posError = detectError(stringIt(demodArr), r)
            print("The redundant bits are " + str(r))
            print("The position of error is " + str(posError))
            print("The size of our data is this " + str(demodArr.size))
            print(str(i) + " out of " + str(EbNo.size))
            decodedData = correctIt(posError, r, demodArr)
            print("")
            '''

            numErrs += np.sum(data != decodedData)
            BERarr[i] = numErrs/decodedData.size
            '''
            f = open("string/hammingCodes.txt", 'w')
            f.write('Corrected Data = ' + str(demodArr))
            f.write('Original Data = ' + str(data))
            f.close()    
            '''
        else:
            modArr = mod.modulate(transArr)
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

