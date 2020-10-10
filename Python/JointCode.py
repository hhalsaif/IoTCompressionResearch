# Importing the tools I need from the commPy library
import commpy.channelcoding
import commpy.utilities

# Importing premade functions that will help with code
from huffman import *
from hamming import *

# huffman code
string = 'BCAADDDCCACACAC'

# Calculating frequency
freq = {}
for c in string:
    if c in freq:
        freq[c] += 1
    else:
        freq[c] = 1

freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
nodes = freq

while len(nodes) > 1:
    (key1, c1) = nodes[-1]
    (key2, c2) = nodes[-2]
    nodes = nodes[:-2]
    node = NodeTree(key1, key2)
    nodes.append((node, c1 + c2))
    nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

huffmanCode = huffman_code_tree(nodes[0][0])

print(freq)
print()
print(' Char | Huffman code ')
print('----------------------')
for (char, frequency) in freq:
    print(' %-4r |%12s' % (char, huffmanCode[char]))

transData = ''
for char in string:
    transData += huffmanCode[char]
print(transData)


print()
# hamming code

# Enter the data to be transmitted
data = transData

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
print("Data transferred is " + arr)

# Stimulate error in transmission by adding gaussiaan noise
transArr = list(arr)
RecieveArr = awgn(arr, snr_dB, rate=1.0)


# important
print("Error Data is " + arr)
correction = hamming_dist(transArr, RecieveArr)
print("The position of error is " + str(correction))
print("")
