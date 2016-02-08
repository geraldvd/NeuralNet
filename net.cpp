#include <iostream>
#include <cassert>
#include <cmath>

#include "net.h"

Net::Net(const std::vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();

    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1 ];


        // We made a new Layer, now fill in ith neurons, and
        // add a bias neuron to the layer:
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            std::cout << "Made a Neuron!" << std::endl;
        }

    }

}

void Net::feedForward(const std::vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1 );

    // Assign (latch) the input values into the input neuros

    for (unsigned i = 0; i < inputVals.size(); ++i )
    {
       m_layers[0][i].setOutputVal(inputVals[i]);

    }
    // Forward propagation
    for(unsigned layerNum = 1; layerNum < m_layers.size() ; ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum - 1];
        for(unsigned neuronNum = 0; neuronNum < m_layers[layerNum].size() - 1; ++neuronNum )
        {
            m_layers[layerNum][neuronNum].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const std::vector<double> &targetVals)
{
    // Calculate the overall net error (RMS)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n )
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;

    }

    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error); // RMS  TODO remove line?

    // Implement a recent average measurement:

    m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);


    // Calculate output layer gradients

    for (unsigned n = 0; n < outputLayer.size() - 1 ; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }


    // Calculate gradients on hidden layers

    for (unsigned layerNum = m_layers.size() - 2 ; layerNum > 0; --layerNum)
    {
        Layer & hiddenLayer = m_layers[layerNum];
        Layer & nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from output to first hidden layers,
    // update conection weight

    for (unsigned layerNum = m_layers.size() - 1 ; layerNum > 0; --layerNum)
    {
        Layer & layer = m_layers[layerNum];
        Layer & prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1 ; ++n)
        {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

std::vector<double> Net::getResults() const
{
    std::vector<double> resultVals;

    const Layer &outputLayer = m_layers.back();

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        resultVals.push_back(outputLayer[n].getOutputVal());
    }

    return resultVals;
}
