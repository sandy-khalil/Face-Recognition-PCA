#include "metrics.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>

using namespace cv;
using namespace cv::ml;
using namespace std;

void printClassificationReport(const Mat& test_labels, const Mat& predictions) {
    int total = test_labels.rows;
    int correct = 0;
    
    map<int, int> true_positives, false_positives, false_negatives, support;
    set<int> classes;

    for (int i = 0; i < total; ++i) {
        int true_label = test_labels.at<int>(i, 0);
        int pred_label = static_cast<int>(predictions.at<float>(i, 0));
        
        classes.insert(true_label);
        classes.insert(pred_label);
        
        support[true_label]++;
        if (true_label == pred_label) {
            correct++;
            true_positives[true_label]++;
        } else {
            false_positives[pred_label]++;
            false_negatives[true_label]++;
        }
    }

    cout << "\n--- Classification Report ---" << endl;
    cout << left << setw(10) << "Class" << setw(12) << "Precision" << setw(12) << "Recall" << setw(12) << "F1-Score" << setw(10) << "Support" << endl;
    cout << string(56, '-') << endl;

    float macro_precision = 0, macro_recall = 0, macro_f1 = 0;

    for (int c : classes) {
        if (support[c] == 0 && false_positives[c] == 0) continue;

        float precision = (true_positives[c] + false_positives[c] > 0) ? (float)true_positives[c] / (true_positives[c] + false_positives[c]) : 0;
        float recall = (support[c] > 0) ? (float)true_positives[c] / support[c] : 0;
        float f1 = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0;

        macro_precision += precision;
        macro_recall += recall;
        macro_f1 += f1;

        cout << left << setw(10) << c 
             << fixed << setprecision(2) << setw(12) << precision 
             << setw(12) << recall 
             << setw(12) << f1 
             << setw(10) << support[c] << endl;
    }

    int n_classes = classes.size();
    cout << string(56, '-') << endl;
    cout << left << setw(34) << "Accuracy" << setw(12) << (float)correct / total << setw(10) << total << endl;
    cout << left << setw(10) << "Macro Avg" 
         << setw(12) << macro_precision / n_classes 
         << setw(12) << macro_recall / n_classes 
         << setw(12) << macro_f1 / n_classes 
         << setw(10) << total << endl;
    cout << "-----------------------------" << endl;
}

void generateROCData(const Mat& train_pca, const Mat& train_labels, const Mat& test_pca, const Mat& test_labels, const string& filename) {
    set<int> classes;
    for (int i = 0; i < train_labels.rows; ++i) classes.insert(train_labels.at<int>(i, 0));

    ofstream file(filename);
    file << "true_label";
    for (int c : classes) file << ",score_" << c;
    file << "\n";

    cout << "Generating One-vs-Rest ROC scores for " << classes.size() << " classes..." << endl;

    for (int i = 0; i < test_pca.rows; ++i) {
        file << test_labels.at<int>(i, 0);
        
        for (int c : classes) {
            // To be efficient, we'd ideally train these once. 
            // But for 40 classes and small data, we can just do it or pre-train.
            // Let's pre-train them for performance if this were a large app.
            // For now, I'll use a static cache or just train them here.
            static map<int, Ptr<SVM>> svm_cache;
            if (svm_cache.find(c) == svm_cache.end()) {
                Mat binary_labels;
                for (int j = 0; j < train_labels.rows; ++j) {
                    binary_labels.push_back(train_labels.at<int>(j, 0) == c ? 1 : 0);
                }
                Ptr<SVM> svm = SVM::create();
                svm->setType(SVM::C_SVC);
                svm->setKernel(SVM::RBF);
                svm->setC(10.0);
                svm->setGamma(0.001);
                svm->train(TrainData::create(train_pca, ROW_SAMPLE, binary_labels));
                svm_cache[c] = svm;
            }
            
            float score = svm_cache[c]->predict(test_pca.row(i), noArray(), StatModel::RAW_OUTPUT);
            file << "," << -score; // Invert because OpenCV's RAW_OUTPUT sign is often flipped for 0/1 labels
        }
        file << "\n";
    }
    file.close();
    cout << "ROC data with scores saved to " << filename << endl;
}

struct ROCPoint { float fpr, tpr; };

vector<ROCPoint> calculateROC(const vector<int>& labels, const vector<float>& scores) {
    int n_pos = 0;
    for (int l : labels) if (l == 1) n_pos++;
    int n_neg = (int)labels.size() - n_pos;

    vector<pair<float, int>> sorted_scores;
    for (size_t i = 0; i < scores.size(); ++i) sorted_scores.push_back({scores[i], labels[i]});
    sort(sorted_scores.rbegin(), sorted_scores.rend());

    vector<ROCPoint> points;
    points.push_back({0, 0});
    int tp = 0, fp = 0;
    for (auto const& p : sorted_scores) {
        if (p.second == 1) tp++;
        else fp++;
        points.push_back({(float)fp / n_neg, (float)tp / n_pos});
    }
    return points;
}

void plotROC(const string& csv_filename, const string& output_filename) {
    ifstream file(csv_filename);
    string line, header;
    if (!getline(file, header)) return;
    
    vector<int> true_labels;
    vector<vector<float>> all_scores;
    vector<int> classes;
    
    stringstream ss_h(header);
    string part;
    getline(ss_h, part, ','); 
    while(getline(ss_h, part, ',')) {
        size_t last_underscore = part.find_last_of('_');
        if (last_underscore != string::npos) {
            classes.push_back(stoi(part.substr(last_underscore + 1)));
            all_scores.push_back({});
        }
    }

    while (getline(file, line)) {
        stringstream ss(line);
        getline(ss, part, ',');
        true_labels.push_back(stoi(part));
        for (size_t i = 0; i < classes.size(); ++i) {
            if (getline(ss, part, ','))
                all_scores[i].push_back(stof(part));
        }
    }

    int size = 600;
    int padding = 60;
    Mat plot(size + padding * 2, size + padding * 2, CV_8UC3, Scalar(255, 255, 255));

    cv::line(plot, Point(padding, padding), Point(padding, size + padding), Scalar(0, 0, 0), 2);
    cv::line(plot, Point(padding, size + padding), Point(size + padding, size + padding), Scalar(0, 0, 0), 2);
    cv::line(plot, Point(padding, size + padding), Point(size + padding, padding), Scalar(200, 200, 200), 1, LINE_AA);

    Scalar colors[] = {Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(0, 165, 255), Scalar(255, 0, 255)};
    
    for (int i = 0; i < min((int)classes.size(), 5); ++i) {
        vector<int> binary_labels;
        for (int l : true_labels) binary_labels.push_back(l == classes[i] ? 1 : 0);
        
        vector<ROCPoint> points = calculateROC(binary_labels, all_scores[i]);
        for (size_t j = 0; j < points.size() - 1; ++j) {
            Point p1(padding + points[j].fpr * size, size + padding - points[j].tpr * size);
            Point p2(padding + points[j+1].fpr * size, size + padding - points[j+1].tpr * size);
            cv::line(plot, p1, p2, colors[i % 5], 2, LINE_AA);
        }
        string label = "Class " + to_string(classes[i]);
        putText(plot, label, Point(size + padding - 100, padding + 30 + i * 25), FONT_HERSHEY_SIMPLEX, 0.5, colors[i % 5], 1);
    }

    putText(plot, "False Positive Rate", Point(size/2, size + padding + 40), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
    putText(plot, "True Positive Rate", Point(10, size/2 + padding), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
    putText(plot, "ROC Curve (C++)", Point(size/2 + padding - 80, padding - 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);

    imwrite(output_filename, plot);
    cout << "ROC Plot saved as " << output_filename << " using C++" << endl;
}
