// Neural net tutorial.cpp

#include <iostream>
#include <utility>
#include <ctime>

#include "net.h"

using namespace std;

typedef std::vector<double> vd;

int main()
{
    srand(time(NULL));

    // e.g., {3, 2, 1} == {3 inputs, 2 hidden neurons (single layer), 1 output}
    vector<unsigned> topology;

    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(1);

    Net myNet(topology);

    // Define training set xor
    vector<pair<vd,vd> > trainingSet;
    {
        vd inputVals;
        inputVals.push_back(0);
        inputVals.push_back(0);
        vd targetVals;
        targetVals.push_back(0);
        trainingSet.push_back(pair<vd,vd>(inputVals, targetVals));
    }
    {
        vd inputVals;
        inputVals.push_back(1);
        inputVals.push_back(0);
        vd targetVals;
        targetVals.push_back(1);
        trainingSet.push_back(pair<vd,vd>(inputVals, targetVals));
    }
    {
        vd inputVals;
        inputVals.push_back(0);
        inputVals.push_back(1);
        vd targetVals;
        targetVals.push_back(1);
        trainingSet.push_back(pair<vd,vd>(inputVals, targetVals));
    }
    {
        vd inputVals;
        inputVals.push_back(1);
        inputVals.push_back(1);
        vd targetVals;
        targetVals.push_back(0);
        trainingSet.push_back(pair<vd,vd>(inputVals, targetVals));
    }

    for(unsigned i=0; i<4; i++) {
        auto &t = trainingSet.at(i);
        cout << t.first.at(0) << ", "<< t.first.at(1) << ", "<< t.second.at(0) << endl;
    }


    // Compose random trainingset (1000 samples)
    vector<pair<vd,vd> > totalTraining;
    for(int N=0; N<10000; N++) {
        int n = (int)(4.0 * rand() / double(RAND_MAX) );
        totalTraining.push_back(trainingSet.at(n));
    }

    // Training
    unsigned counter{0};
    for(auto &t : totalTraining) {
        myNet.feedForward(t.first);
        cout << "Pass " << ++counter << ": Inputs: " << t.first.at(0) << ' ' << t.first.at(1) << endl;
        cout << "Output: " << myNet.getResults().at(0) << endl;
        cout << "Target: " << t.second.at(0) << endl;
        myNet.backProp(t.second);
        cout << "Net recent average error: " << myNet.getRecentAverageError() << endl << endl;
    }


    // Testing
    cout << "Testing: " << endl;
    cout << "x1\tx2\ty" << endl;
    for(double i=0.0; i<=1; i+= 0.5) {
        for(double j=0.0; j<=1; j+= 0.5) {
            vd input{i,j};
            myNet.feedForward(input);

            cout << input.at(0) << '\t' << input.at(1) << '\t' << myNet.getResults().at(0) << endl;
        }
    }

    return 0;
}

