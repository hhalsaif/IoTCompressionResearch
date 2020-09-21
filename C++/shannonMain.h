#pragma once
#include <iostream>
#include <vector>
using namespace std;

int noOfBits = 10;
vector<int> dataTrans;
// declare structure node
struct node 
{
    // for storing symbol
    string sym;

    // for stroing probability or frequency
    float pro;
    int* arr = new int [noOfBits];
    int top;
}p[noOfBits];

typedef struct node node;

//function to find shannon code
void shannon(int l, int h, vector<node> p)
{
    float pack1 = 0, pack2 = 0;
    float diff1 = 0, diff2 = 0;
    int d, k;

    if  ((l + 1) == h || 1 == h || 1 > h)
    {
        if (l == h || 1 >h)
            return;
        
        p[h].arr[++(p[h].top)] = 0;
        p[l].arr[++(p[l].top)] = 0;
        return;
    }

    else 
    {
        for (int i = l; i <= h - 1; i++)
            pack1 = pack1 + p[i].pro;
        
        pack2 = pack2 + p[h].pro;
        diff1 = pack1 - pack2; // finding the first difference between the two packets

        if(diff1 < 0)
            diff1 = diff1 * -1; // making sure the answer is not negative
        
        int j = 2;
        while (j != h -1 + l)
        {
            k = h - j;
            pack1 = pack2 = 0;

            for (int i = l; i <= k; i++)
                pack1 = pack1 + p[i].pro;
            for (int i = h; i > k; i--)
                pack2 = pack2 + p[i].pro;
            
            diff2 = pack1 - pack2; // finding the second difference between the two lackets

            if (diff2 < 0)
                diff2 = diff2 * -1; //making sure the answer is not negative
            if (diff2 >= diff1)
                break;
            
            diff1 = diff2;
            j++;
        }
        
        k++;
        for (int i = l; i <= k; i++)
            p[i].arr[++(p[i].top)] = 1;
        for (int i = k + 1; i <= h; i++)
            p[i].arr[++(p[i].top)] = 0;
        
        //Invoke the function
        shannon(l, k, p);
        shannon(k + 1, h, p);
    }
}

// Function to sort the symbols 
// based on their probability or frequency 
void sortByProbability(int numBits, node p[]) 
{ 
	node temp; 
	for (int j = 1; j <= numBits - 1; j++) { 
		for (int i = 0; i < numBits - 1; i++) { 
			if ((p[i].pro) > (p[i + 1].pro)) { 
				temp.pro = p[i].pro; 
				temp.sym = p[i].sym; 

				p[i].pro = p[i + 1].pro; 
				p[i].sym = p[i + 1].sym; 

				p[i + 1].pro = temp.pro; 
				p[i + 1].sym = temp.sym; 
			} 
		} 
	} 
}

// function to display shannon codes 
void display(int numBits, node p[]) 
{ 
	int i, j; 
	cout << "\n\n\n\tSymbol\tProbability\tCode"; 
	for (i = numBits - 1; i >= 0; i--) { 
		cout << "\n\t" << p[i].sym << "\t\t" << p[i].pro << "\t"; 
		for (j = 0; j <= p[i].top; j++) 
			cout << p[i].arr[j]; 
	} 
} 

void ShannonDriver(vector<int> transData)
{
    noOfBits = transData.size();
    float total = 0;
    vector<float> bitProb;
    string ch;
    node temp;

    srand(0);

    for(int i = 0; i < noOfBits; i++)
    {
        ch = (char)(65+i);

        // Insert the symbol to node
        p[i].sym += ch;
    }

    // creating probability
    int fullBit = 1;
    while(fullBit > 0)
    {
        float symProbability = (rand() % 100) / 100;
        fullBit -= symProbability;

        if(fullBit < 0)
            fullBit += symProbability;
        
        else 
              bitProb.push_back(symProbability); 
        
    }

    // Input probability of symbols 
    for(int i = 0; i < noOfBits; i++)
    {
        // Insert the value to node
        p[i].pro = bitProb[i];
        total += p[i].pro;

        //checking max probability
        if(total > 1)
        {
            cout << "Invalid";
            total -= p[i].pro;
            i--;
        }
    }

    p[noOfBits].pro = 1 - total;

    sortByProbability(noOfBits, p);

    for(int i = 0; i < noOfBits; i++)
        p[i].top = -1;
    
    // Find the shannon code
    shannon(0, noOfBits - 1, p);

    // Display the codes
    display(noOfBits, p);
    return;
}

