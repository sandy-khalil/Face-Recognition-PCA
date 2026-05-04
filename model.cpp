#include "model.h"
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

void applyPCA(const Mat& train_images, int n_components, CustomPCA& pca, Mat& train_pca) {
    // 1. Calculate Mean
    reduce(train_images, pca.mean, 0, REDUCE_AVG);

    // 2. Center the data using vectorized subtraction
    Mat centered;
    subtract(train_images, repeat(pca.mean, train_images.rows, 1), centered);

    // 3. Compute Covariance Matrix of the dual space (X * X^T)
    // This is N x N (320x320) instead of D x D (10304x10304)
    Mat cov = (centered * centered.t()) / (train_images.rows - 1);

    // 4. Eigendecomposition of the smaller matrix
    Mat eigenvalues, eigenvectors_v;
    eigen(cov, eigenvalues, eigenvectors_v);

    // 5. Project back to get eigenvectors of the original space: u = X^T * v
    // We only need the top n_components
    int actual_components = min(n_components, centered.rows);
    Mat top_eigenvalues = eigenvalues.rowRange(0, actual_components);
    Mat top_v = eigenvectors_v.rowRange(0, actual_components);
    
    pca.eigenvalues = top_eigenvalues.clone();
    pca.eigenvectors = top_v * centered;

    // 6. Normalize eigenvectors in the original space
    for (int i = 0; i < pca.eigenvectors.rows; ++i) {
        normalize(pca.eigenvectors.row(i), pca.eigenvectors.row(i));
    }

    // 7. Project training data
    train_pca = centered * pca.eigenvectors.t();

    // 8. Manual Whitening
    for (int i = 0; i < train_pca.rows; ++i) {
        for (int j = 0; j < train_pca.cols; ++j) {
            float ev = pca.eigenvalues.at<float>(j);
            if (ev > 1e-5) train_pca.at<float>(i, j) /= sqrt(ev);
        }
    }
}

Ptr<SVM> trainSVM(const Mat& train_pca, const Mat& train_labels) {
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setC(10.0);
    svm->setGamma(0.001);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    
    Ptr<TrainData> trainData = TrainData::create(train_pca, ROW_SAMPLE, train_labels);
    svm->train(trainData);
    
    return svm;
}
