// C++ program for Shannon Fano Algorithm 

// include header files 
#include <bits/stdc++.h> 
using namespace std; 

int sizeOfData;

// declare structure node 
struct node { 

	// for storing symbol 
	string sym; 

	// for storing probability or frquency 
	float pro; 
	int *arr = new int[sizeOfData]; 
	int top; 
} p[sizeOfData]; //this probably needs to be fixed 

typedef struct node node; 

// function to find shannon code 
void shannon(int l, int h, node p[]) 
{ 
	float pack1 = 0, pack2 = 0, diff1 = 0, diff2 = 0; 
	int i, d, k, j; 
	if ((l + 1) == h || l == h || l > h) { 
		if (l == h || l > h) 
			return; 
		p[h].arr[++(p[h].top)] = 0; 
		p[l].arr[++(p[l].top)] = 1; 
		return; 
	} 
	else { 
		for (i = l; i <= h - 1; i++) 
			pack1 = pack1 + p[i].pro; 
		pack2 = pack2 + p[h].pro; 
		diff1 = pack1 - pack2; 
		if (diff1 < 0) 
			diff1 = diff1 * -1; 
		j = 2; 
		while (j != h - l + 1) { 
			k = h - j; 
			pack1 = pack2 = 0; 
			for (i = l; i <= k; i++) 
				pack1 = pack1 + p[i].pro; 
			for (i = h; i > k; i--) 
				pack2 = pack2 + p[i].pro; 
			diff2 = pack1 - pack2; 
			if (diff2 < 0) 
				diff2 = diff2 * -1; 
			if (diff2 >= diff1) 
				break; 
			diff1 = diff2; 
			j++; 
		} 
		k++; 
		for (i = l; i <= k; i++) 
			p[i].arr[++(p[i].top)] = 1; 
		for (i = k + 1; i <= h; i++) 
			p[i].arr[++(p[i].top)] = 0; 

		// Invoke shannon function 
		shannon(l, k, p); 
		shannon(k + 1, h, p); 
	} 
} 

// Function to sort the symbols 
// based on their probability or frequency 
void sortByProbability(int n, node p[]) 
{ 
	int i, j; 
	node temp; 
	for (j = 1; j <= n - 1; j++) { 
		for (i = 0; i < n - 1; i++) { 
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
void display(int n, node p[]) 
{ 
	int i, j; 
	cout << "\n\n\n\tSymbol\tProbability\tCode"; 
	for (i = n - 1; i >= 0; i--) { 
		cout << "\n\t" << p[i].sym << "\t\t" << p[i].pro << "\t"; 
		for (j = 0; j <= p[i].top; j++) 
			cout << p[i].arr[j]; 
	} 
} 

// Driver code 
vector<int> shannonDriver(vector<int> transData) 
{
    int n;
    float total = 0;
    string ch;
    node temp;
    sizeOfData = transData.size();

    // Input symbols
    for(int i = 0; i < transData.size(); i++)
    {
        ch = (char)(65 + i);

        // Insert the symbol to node
        p[i].sym += ch;
    }
    // Input probability of symbols 
	float x[] = { 0.22, 0.28, 0.15, 0.30, 0.05 }; 
	int i;
    for (i = 0; i < transData.size(); i++) { 
		

		// Insert the value to node 
		p[i].pro = x[i]; 
		total = total + p[i].pro; 

		// checking max probability 
		if (total > 1) { 
			cout << "Invalid. Enter new values"; 
			total = total - p[i].pro; 
			i--; 
		} 
	} 

    p[i].pro = 1 - total; 

    // Sorting the symbols based on 
	// their probability or frequency 
	sortByProbability(transData.size(), p); 

	for (i = 0; i < transData.size(); i++) 
		p[i].top = -1; 

    // Find the shannon code 
	shannon(0, transData.size() - 1, p); 

	// Display the codes 
	display(transData.size(), p); 
	
    vector<int> compData;
    for(int i = 0; i < sizeof(p); i++)
        compData.push_back(p[i]);
    
    // The topPart doesnt really work

    return compData;
}
