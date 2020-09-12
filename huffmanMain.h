#pragma once 
#include <iostream>
#include <cstdlib>
using namespace std;

// This constant can be avoided by explicitly 
// calculating height of Huffman Tree 
#define MAX_TREE_HT 100 

// A Huffman tree node 
struct MinHeapNode {

	// One of the input characters 
	char data;

	// Frequency of the character 
	unsigned freq;

	// Left and right child of this node 
	struct MinHeapNode* left, * right;
};

// A Min Heap: Collection of 
// min-heap (or Huffman tree) nodes 
struct MinHeap {

	// Current size of min heap 
	unsigned size;

	// capacity of min heap 
	unsigned capacity;

	// Attay of minheap node pointers 
	struct MinHeapNode** array;
};

// A utility function allocate a new 
// min heap node with given character 
// and frequency of the character 
struct MinHeapNode* newNode(char data, unsigned freq)
{
	struct MinHeapNode* temp
		= (struct MinHeapNode*)malloc
		(sizeof(struct MinHeapNode));

	temp->left = temp->right = NULL;
	temp->data = data;
	temp->freq = freq;

	return temp;
}

// A utility function to create 
// a min heap of given capacity 
struct MinHeap* createMinHeap(unsigned capacity)

{

	struct MinHeap* minHeap
		= (struct MinHeap*)malloc(sizeof(struct MinHeap));

	// current size is 0 
	minHeap->size = 0;

	minHeap->capacity = capacity;

	minHeap->array
		= (struct MinHeapNode**)malloc(minHeap->
			capacity * sizeof(struct MinHeapNode*));
	return minHeap;
}

void swapMinHeapNode(struct MinHeapNode** a,
	struct MinHeapNode** b)

{

	struct MinHeapNode* t = *a;
	*a = *b;
	*b = t;
}

// The standard minHeapify function. 
void minHeapify(struct MinHeap* minHeap, int idx)

{

	int smallest = idx;
	int left = 2 * idx + 1;
	int right = 2 * idx + 2;

	if (left < minHeap->size && minHeap->array[left]->
		freq < minHeap->array[smallest]->freq)
		smallest = left;

	if (right < minHeap->size && minHeap->array[right]->
		freq < minHeap->array[smallest]->freq)
		smallest = right;

	if (smallest != idx) {
		swapMinHeapNode(&minHeap->array[smallest],
			&minHeap->array[idx]);
		minHeapify(minHeap, smallest);
	}
}

// A utility function to check 
// if size of heap is 1 or not 
int isSizeOne(struct MinHeap* minHeap)
{

	return (minHeap->size == 1);
}


