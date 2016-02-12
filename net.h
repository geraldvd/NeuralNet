#ifndef NET_H
#define NET_H

#include<vector>

#include "neuron.h"



class Net
{
public:

    Net(const std::vector<unsigned> &topology );

    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);        // training function
    std::vector<double> getResults() const;

    double getRecentAverageError() const;
    void setRecentAverageError(const double &recentAverageError);

private:
    std::vector<Layer> m_layers; // m_layers[layerNum][neuroNum]
    double m_error;

    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;

};

#endif // NET_H
