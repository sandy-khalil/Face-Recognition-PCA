#ifndef METRICS_H
#define METRICS_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>

/**
 * Prints a classification report including accuracy and per-class precision/recall.
 */
void printClassificationReport(const cv::Mat& test_labels, const cv::Mat& predictions);

/**
 * Generates ROC data using One-vs-Rest classification to provide real scores.
 */
void generateROCData(const cv::Mat& train_pca, const cv::Mat& train_labels, 
                    const cv::Mat& test_pca, const cv::Mat& test_labels, 
                    const std::string& filename);

/**
 * Draws the ROC curve onto an OpenCV Mat and saves it to a file.
 */
void plotROC(const std::string& csv_filename, const std::string& output_filename);

#endif
