// Neural net tutorial.cpp

#include <iostream>

#include "net.h"

using namespace std;


int main()
{
    // e.g., {3, 2, 1}
    std::vector<unsigned> topology;

    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);

    Net myNet(topology);



    std::vector<double> inputVals;

    inputVals.push_back(0.3);
    inputVals.push_back(0.5);
    inputVals.push_back(0.7);

    myNet.feedForward(inputVals);

    std::vector<double> targetVals;

    targetVals.push_back(0.3/2);
    targetVals.push_back(0.5/2);
    targetVals.push_back(0.7/2);

    myNet.backProp(targetVals);

    //std::vector<double> resultVals = myNet.getResults();


}

