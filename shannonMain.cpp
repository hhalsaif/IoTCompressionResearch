#pragma once
#include <bits/stdc++.h>
#include <vector>
using namespace std;

int noOfBits = 10;

// declare structure node
struct node 
{
    // for storing symbol
    string sym;

    // for stroing probability or frequency
    float pro;
    vector<int> arr[noOfBits];
    int top;
}p[noOfBits];

void assignMain(int noBit)
{
    noOfBits = noBit;
}

typedef struct node node;

//function to find shannong code
void shannon(int l, int h, node p[])
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

