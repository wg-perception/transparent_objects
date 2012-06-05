#ifndef NONMAXIMUMSUPPRESSION_HPP
#define NONMAXIMUMSUPPRESSION_HPP

#include <opencv2/core/core.hpp>

void filterOutLowValues(const std::vector<float> &values, float ratioToMaximum,
                        std::vector<bool> &isFilteredOut);
void filterOutHighValues(const std::vector<float> &values, float ratioToMinimum,
                         std::vector<bool> &isFilteredOut);

void filterOutNonMaxima(const std::vector<float> &values, const std::vector<std::vector<int> > &neighbors,
                        std::vector<bool> &isFilteredOut);
void filterOutNonMinima(const std::vector<float> &values, const std::vector<std::vector<int> > &neighbors,
                        std::vector<bool> &isFilteredOut);

void suppressNonMaxima(std::vector<float> &values, const std::vector<std::vector<int> > &neighbors, float ratioToMaximum);

template<class T>
void filterValues(std::vector<T> &values, const std::vector<bool> &isFilteredOut)
{
  CV_Assert(values.size() == isFilteredOut.size());

  std::vector<T> filteredValues;
  for (size_t i = 0; i < values.size(); ++i)
  {
    if (!isFilteredOut[i])
    {
      filteredValues.push_back(values[i]);
    }
  }

  std::swap(values, filteredValues);
}

#endif // NONMAXIMUMSUPPRESSION_HPP
