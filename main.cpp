#include <iostream>
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
#include <vector>
#include "shannonMain.h"
#include "huffman.h"
using namespace std; 

vector<int> transData;
int numOfBits = 10;

void createData ()
{
    srand(time(NULL));
    int numOfBits = rand() % 10 + 7;
    for (int i = 0; i < numOfBits; i++)
    {
        int bit = rand() % 1;
        transData.push_back(bit);
    }
}

int main ()
{
    createData();
    //ShannonDriver(transData);
    huffmanDriver(transData);
}