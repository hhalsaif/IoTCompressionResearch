#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
#include <vector>
#include <shannonMain.cpp>
using namespace std; 

vector<int> data;
int numOfBits = 10;

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
    assignMain(numOfBits);
}