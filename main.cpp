#include <bits/stdc++.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
#include <vector>
using namespace std; 

vector<int> data;
int numOfBits = 10;

// declare structure node 
struct node { 

	// for storing symbol 
	string sym; 

	// for storing probability or frquency 
	float pro; 
    vector<int> arr;
	int top; 
} /* p[20] its really p[numOfBits] but it needs to be a constant.... */;

typedef struct node node; 

void createData ()
{
    srand(time(NULL));
    int numOfBits = rand() % 10 + 7;
    for (int i = 0; i < numOfBits; i++)
    {
        int bit = rand() % 1;
        data.push_back(bit);
    }
}

int main ()
{
    createData();
}