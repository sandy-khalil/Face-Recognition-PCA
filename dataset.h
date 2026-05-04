#ifndef DATASET_H
#define DATASET_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * Loads the AT&T face dataset.
 * Detects faces using Haar Cascade, resizes to 92x112, and flattens them.
 */
void loadDataset(const std::string& path, cv::Mat& images, cv::Mat& labels);

/**
 * Splits the dataset into training and testing sets.
 */
void splitDataset(const cv::Mat& images, const cv::Mat& labels, float train_ratio, 
                  cv::Mat& train_images, cv::Mat& train_labels, 
                  cv::Mat& test_images, cv::Mat& test_labels);

#endif
