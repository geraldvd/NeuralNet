#include <cstdlib>
#include <cmath>

#include "neuron.h"


double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(const unsigned &numOutputs, const unsigned &myIndex)
{
    for(unsigned outputNum = 0; outputNum < numOutputs; ++outputNum)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();

    }

m_myIndex = myIndex;

}

double Neuron::randomWeight()
{
    return rand() / double(RAND_MAX);
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

double Neuron::getOutputVal() const
{
    return m_outputVal;
}

void Neuron::setOutputVal(const double &outputVal)
{
    m_outputVal = outputVal;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    // Sum the prevous layer's output (which are our inputs)
    // Include the bias node from previous layer

    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = transferFunction(sum);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(const double &targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neuron in the precending layer

    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the gradient train rate
                eta
                * neuron_getOutputVal()  // TODO Solve error
                * m_gradient
                // Also add momentum = a fraction of the previous delta weight
                * alpha
                * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;

    }
}

double Neuron::transferFunction(const double &x)
{
    // tanh - output range [ -1.0 , 1.0 ]
    return tanh(x);
}

double Neuron::transferFunctionDerivative(const double &x)
{
    return 1 - tanh(x) * tanh(x); // TODO return 1 - x*x;
}

