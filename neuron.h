#ifndef NEURON_H
#define NEURON_H

#include<vector>

#include "connection.h"

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(const unsigned &numOutputs, const unsigned &myIndex);

    double getOutputVal() const;
    void setOutputVal(const double &getOutputVal);
    void feedForward(const Layer &prevLayer);

    void calcHiddenGradients(const Layer &nextLayer);
    void calcOutputGradients(const double &targetVal);

    void updateInputWeights(Layer &prevLayer);

private:
    static double eta; // [0.0 ..... 1.0] overall net training rate
    static double alpha; // [0.0 , ..... , n] multiplier of last weight change [momentum]

    unsigned m_myIndex;

    static double transferFunction(const double &x);
    static double transferFunctionDerivative(const double &x);
    static double randomWeight(void);
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    std::vector<Connection> m_outputWeights;
    double m_gradient;


};



#endif // NEURON_H
