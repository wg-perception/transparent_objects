#include <algorithm>
#include <iostream>
#include "edges_pose_refiner/nonMaximumSuppression.hpp"

using std::vector;
using std::cout;
using std::endl;

void filterOutLowValues(const std::vector<float> &values, float ratioToMaximum,
                        std::vector<bool> &isFilteredOut)
{
  if (values.empty())
  {
    isFilteredOut.clear();
    return;
  }

//  CV_Assert(ratioToMaximum < 1.0f);

  if (isFilteredOut.empty())
  {
    isFilteredOut.resize(values.size(), false);
  }
  else
  {
    CV_Assert(isFilteredOut.size() == values.size());
  }

  float maxValue = *std::max_element(values.begin(), values.end());

  for (size_t i = 0; i < values.size(); ++i)
  {
    isFilteredOut[i] = isFilteredOut[i] || (values[i] < ratioToMaximum * maxValue);
  }
}

void filterOutHighValues(const std::vector<float> &values, float ratioToMinimum,
                         std::vector<bool> &isFilteredOut)
{
  std::vector<float> negativeValues(values.size());
  for (size_t i = 0; i < values.size(); ++i)
  {
    negativeValues[i] = -values[i];
  }

  filterOutLowValues(negativeValues, ratioToMinimum, isFilteredOut);
}

void filterOutNonMaxima(const std::vector<float> &values, const std::vector<std::vector<int> > &neighbors,
                        std::vector<bool> &isFilteredOut)
{
  if (values.empty())
  {
    isFilteredOut.clear();
    return;
  }

  CV_Assert(values.size() == neighbors.size());
  if (isFilteredOut.empty())
  {
    isFilteredOut.resize(values.size(), false);
  }
  else
  {
    CV_Assert(isFilteredOut.size() == values.size());
  }

  for (size_t valueIndex = 0; valueIndex < values.size(); ++valueIndex)
  {
    if (isFilteredOut[valueIndex])
    {
      continue;
    }

    for (size_t neighborIndex = 0; neighborIndex < neighbors[valueIndex].size(); ++neighborIndex)
    {
      if (values[valueIndex] < values[neighbors[valueIndex][neighborIndex]])
      {
        isFilteredOut[valueIndex] = true;
        break;
      }
    }
  }
}

void filterOutNonMinima(const std::vector<float> &values, const std::vector<std::vector<int> > &neighbors,
                        std::vector<bool> &isFilteredOut)
{
  std::vector<float> negativeValues(values.size());
  for (size_t i = 0; i < values.size(); ++i)
  {
    negativeValues[i] = -values[i];
  }

  filterOutNonMaxima(negativeValues, neighbors, isFilteredOut);
}

void suppressNonMaxima(std::vector<float> &values, const std::vector<std::vector<int> > &neighbors, float ratioToMaximum)
{
  vector<bool> isFilteredOut;
  filterOutLowValues(values, ratioToMaximum, isFilteredOut);
  filterOutNonMaxima(values, neighbors, isFilteredOut);

  vector<float> filteredValues;
  for (size_t i = 0; i < isFilteredOut.size(); ++i)
  {
    if (!isFilteredOut[i])
    {
      filteredValues.push_back(values[i]);
    }
  }

  std::swap(values, filteredValues);
}
