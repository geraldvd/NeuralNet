// Neural net tutorial.cpp

#include <iostream>
#include <utility>
#include <ctime>
#include <cmath>

#include "net.h"

using namespace std;

typedef std::vector<double> vd;

int main2()
{
    // Set RNG seed
    srand(time(NULL));

    // e.g., {3, 2, 1} == {3 inputs, 2 hidden neurons (single layer), 1 output}
    vector<unsigned> topology;

    topology.push_back(1);
    topology.push_back(4);
    topology.push_back(4);
    topology.push_back(1);

    Net myNet(topology);


    // Compose random trainingset
//    cout << "x\ty" << endl;
    vector<pair<vd,vd> > totalTraining;
    for(int N=0; N<1000000; N++) {
        vd x = {rand() / double(RAND_MAX)};
        vd y = {sin(2*M_PI*x.at(0))};
        totalTraining.push_back(pair<vd,vd>(x,y));
//        cout << x.at(0) << '\t' << y.at(0) << endl;
    }

    // Training
    unsigned counter{0};
    for(auto &t : totalTraining) {
        myNet.feedForward(t.first);
//        cout << "Pass " << ++counter << ": Input: " << t.first.at(0) << endl;
//        cout << "Output: " << myNet.getResults().at(0) << endl;
//        cout << "Target: " << t.second.at(0) << endl;
        myNet.backProp(t.second);
//        cout << "Net recent average error: " << myNet.getRecentAverageError() << endl << endl;
    }


    // Testing
    cout << "Testing: " << endl;
    cout << "x\ty\tyd" << endl;
    for(double x=0.0; x<=2*M_PI; x+= 0.1) {
        vd input{x/2/M_PI};
        myNet.feedForward(input);

        cout << input.at(0) << '\t' << myNet.getResults().at(0) << '\t' << sin(x) << endl;
    }

    return 0;
}


int main()
{
    // Set RNG seed
    srand(time(NULL));

    // e.g., {3, 2, 1} == {3 inputs, 2 hidden neurons (single layer), 1 output}
    vector<unsigned> topology;

    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(4);
    topology.push_back(1);

    Net myNet(topology);


    // Compose random trainingset
//    cout << "x\ty" << endl;
    vector<pair<vd,vd> > totalTraining;
    for(int N=0; N<1000000; N++) {
        vd x = {rand() / double(RAND_MAX), rand() / double(RAND_MAX)};
        vd y = {(2*M_PI*x[0]*2*sin(x[1] * 2*M_PI) + cos(x[0] * 2*M_PI)) / 15};
        totalTraining.push_back(pair<vd,vd>(x,y));
//        cout << x.at(0) << '\t' << y.at(0) << endl;
    }

    // Training
//    unsigned counter{0};
    for(auto &t : totalTraining) {
        myNet.feedForward(t.first);
//        cout << "Pass " << ++counter << ": Input: " << t.first.at(0) << endl;
//        cout << "Output: " << myNet.getResults().at(0) << endl;
//        cout << "Target: " << t.second.at(0) << endl;
        myNet.backProp(t.second);
//        cout << "Net recent average error: " << myNet.getRecentAverageError() << endl << endl;
    }


    // Testing
    cout << "Testing: " << endl;
    cout << "x1\tx2\ty\tyd" << endl;
    for(double x1=0.0; x1<=2*M_PI; x1+=0.2) {
        for(double x2=0.0; x2<=2*M_PI; x2+=0.2) {
            vd input{x1/2/M_PI, x2/2/M_PI};
            myNet.feedForward(input);

            cout << input.at(0) << '\t' << input.at(1) << '\t' << myNet.getResults().at(0)
                 << '\t' << (x1*2*sin(x2) + cos(x1)) / 15 << endl;
        }
    }

    return 0;
}


