#ifndef MODEL_H
#define MODEL_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

/**
 * Custom PCA structure to store manual implementation details.
 */
struct CustomPCA {
    cv::Mat mean;
    cv::Mat eigenvectors;
    cv::Mat eigenvalues;
    
    cv::Mat project(const cv::Mat& data) const {
        cv::Mat centered;
        cv::subtract(data, cv::repeat(mean, data.rows, 1), centered);
        return centered * eigenvectors.t();
    }
};

/**
 * Applies PCA (from scratch) to the training images and projects them.
 */
void applyPCA(const cv::Mat& train_images, int n_components, CustomPCA& pca, cv::Mat& train_pca);

/**
 * Trains an SVM classifier with the provided PCA-transformed data.
 */
cv::Ptr<cv::ml::SVM> trainSVM(const cv::Mat& train_pca, const cv::Mat& train_labels);

#endif
