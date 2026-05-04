#include "dataset.h"
#include "model.h"
#include "metrics.h"
#include "libs/httplib.h"
#include "libs/json.hpp"
#include <iostream>
#include <string>
#include <dirent.h>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;
using namespace httplib;
using json = nlohmann::json;

// Global state to persist model between web requests
struct GlobalState {
    CustomPCA pca;
    cv::Ptr<cv::ml::SVM> svm;
    Mat train_pca;
    Mat train_labels;
    bool trained = false;
    string dataset_path = "att_faces";
} g_state;

// Forward declarations
json runFullAnalysis(string dataset_path, int n_components, float train_ratio, bool json_output, bool include_roc = true);
json predictSingleImage(string image_path, bool json_output);
void startServer();

int main(int argc, char** argv) {
    // If no arguments, start the web server
    if (argc == 1) {
        startServer();
        return 0;
    }

    // Default parameters for CLI
    string dataset_path = "att_faces";
    int n_components = 80;
    float train_ratio = 0.8f;
    bool json_output = false;
    string predict_img = "";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--json") {
            json_output = true;
        } else if (arg == "--predict" && i + 1 < argc) {
            predict_img = argv[++i];
        } else if (i == 1 && arg[0] != '-') {
            dataset_path = arg;
        } else if (i == 2 && arg[0] != '-') {
            n_components = stoi(arg);
        } else if (i == 3 && arg[0] != '-') {
            train_ratio = stof(arg);
        }
    }

    if (!predict_img.empty()) {
        // We need to have a trained model first. For CLI, we'll train on the fly if not trained.
        if (!g_state.trained) {
            runFullAnalysis(dataset_path, n_components, train_ratio, true);
        }
        json res = predictSingleImage(predict_img, json_output);
        if (json_output) cout << res.dump() << endl;
        return 0;
    }

    json res = runFullAnalysis(dataset_path, n_components, train_ratio, json_output);
    if (json_output) cout << res.dump() << endl;

    return 0;
}

json runFullAnalysis(string dataset_path, int n_components, float train_ratio, bool json_output, bool include_roc) {
    if (!json_output) {
        cout << "Dataset Path: " << dataset_path << endl;
        cout << "PCA Components: " << n_components << endl;
        cout << "Train Ratio: " << train_ratio << endl;
    }

    // Check if dataset path exists, if not try ../path (common for build directories)
    DIR* dir = opendir(dataset_path.c_str());
    if (!dir) {
        string fallback_path = "../" + dataset_path;
        DIR* fallback_dir = opendir(fallback_path.c_str());
        if (fallback_dir) {
            if (!json_output) cout << "Notice: Dataset not found at '" << dataset_path << "', using fallback: '" << fallback_path << "'" << endl;
            dataset_path = fallback_path;
            closedir(fallback_dir);
        }
    } else {
        closedir(dir);
    }
    g_state.dataset_path = dataset_path;

    // 1. Load Dataset
    Mat images;
    Mat labels(0, 1, CV_32S);
    if (!json_output) cout << "\nLoading images and detecting faces..." << endl;
    loadDataset(dataset_path, images, labels);

    if (images.empty()) {
        cerr << "Error: No images loaded. Check the dataset path." << endl;
        return {{"error", "No images loaded"}};
    }
    if (!json_output) cout << "Loaded " << images.rows << " images." << endl;

    // 2. Split Dataset
    Mat train_images, train_labels, test_images, test_labels;
    splitDataset(images, labels, train_ratio, train_images, train_labels, test_images, test_labels);
    if (!json_output) cout << "Training set: " << train_images.rows << ", Test set: " << test_images.rows << endl;

    // 3. PCA Extraction
    if (!json_output) cout << "\nApplying PCA (Manual Implementation)..." << endl;
    applyPCA(train_images, n_components, g_state.pca, g_state.train_pca);
    g_state.train_labels = train_labels;

    // 4. Project Test Data
    Mat test_pca = g_state.pca.project(test_images);
    // Apply whitening to test data using the same eigenvalues
    Mat eigenvalues = g_state.pca.eigenvalues;
    for (int i = 0; i < test_pca.rows; ++i) {
        for (int j = 0; j < test_pca.cols; ++j) {
            float ev = eigenvalues.at<float>(j);
            if (ev > 1e-5) test_pca.at<float>(i, j) /= sqrt(ev);
        }
    }

    // 5. Train SVM
    if (!json_output) cout << "Training SVM Classifier..." << endl;
    g_state.svm = trainSVM(g_state.train_pca, g_state.train_labels);
    g_state.trained = true;

    // 6. Predict
    Mat predictions;
    g_state.svm->predict(test_pca, predictions);

    // 7. Evaluate and Report
    int correct = 0;
    for (int i = 0; i < predictions.rows; ++i) {
        if (predictions.at<float>(i) == test_labels.at<int>(i)) correct++;
    }
    float accuracy = (float)correct / predictions.rows;

    if (!json_output) {
        printClassificationReport(test_labels, predictions);
        if (include_roc) {
            generateROCData(g_state.train_pca, g_state.train_labels, test_pca, test_labels, "roc_data.csv");
            plotROC("roc_data.csv", "roc_curve_cpp.png");
        }
    }

    return {
        {"accuracy", accuracy},
        {"train_size", train_images.rows},
        {"test_size", test_images.rows}
    };
}

json predictSingleImage(string image_path, bool json_output) {
    if (!g_state.trained) {
        return {{"error", "Model not trained"}};
    }

    Mat img = imread(image_path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        return {{"error", "Could not load image"}};
    }

    // Use the same face detection as in loading
    CascadeClassifier face_cascade;
    string cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    if (!face_cascade.load(cascade_path)) {
        return {{"error", "Could not load face cascade"}};
    }

    vector<Rect> faces;
    face_cascade.detectMultiScale(img, faces, 1.1, 4);
    
    if (faces.empty()) {
        return {{"predicted_label", -1}, {"message", "No face detected"}};
    }

    // Use the first detected face
    Mat face_img = img(faces[0]);
    resize(face_img, face_img, Size(92, 112));
    
    Mat flattened;
    face_img.reshape(1, 1).convertTo(flattened, CV_32F);
    Mat projected = g_state.pca.project(flattened);
    
    // Whitening
    for (int j = 0; j < projected.cols; ++j) {
        float ev = g_state.pca.eigenvalues.at<float>(j);
        if (ev > 1e-5) projected.at<float>(0, j) /= sqrt(ev);
    }

    float label = g_state.svm->predict(projected);
    
    // --- Unknown Subject Detection (Distance Threshold) ---
    double min_dist = -1;
    for (int i = 0; i < g_state.train_pca.rows; ++i) {
        double dist = norm(projected, g_state.train_pca.row(i), NORM_L2);
        if (min_dist < 0 || dist < min_dist) {
            min_dist = dist;
        }
    }

    bool unknown = (min_dist > 12.0); 

    if (unknown) {
        return {{"predicted_label", -2}, {"message", "Unknown Subject"}, {"distance", min_dist}};
    } else {
        return {{"predicted_label", (int)label}, {"distance", min_dist}};
    }
}

void startServer() {
    Server svr;

    // Resolve 'web' directory path
    string web_path = "web";
    DIR* dir = opendir(web_path.c_str());
    if (!dir) {
        string fallback = "../web";
        DIR* fallback_dir = opendir(fallback.c_str());
        if (fallback_dir) {
            web_path = fallback;
            closedir(fallback_dir);
        }
    } else {
        closedir(dir);
    }

    // Serve static files from the found 'web' directory
    if (!svr.set_mount_point("/", web_path)) {
        cerr << "Error: Could not mount 'web' directory at " << web_path << endl;
        return;
    }

    // Endpoint to run analysis
    svr.Get("/run", [&](const Request& req, Response& res) {
        int n_components = req.has_param("n_components") ? stoi(req.get_param_value("n_components")) : 80;
        float train_ratio = req.has_param("train_ratio") ? stof(req.get_param_value("train_ratio")) : 0.8f;

        try {
            bool include_roc = !req.has_param("include_roc") || req.get_param_value("include_roc") == "true";
            json result = runFullAnalysis("att_faces", n_components, train_ratio, true, include_roc);
            res.set_content(result.dump(), "application/json");
        } catch (const exception& e) {
            res.status = 500;
            res.set_content(e.what(), "text/plain");
        }
    });

    // Endpoint to handle image prediction
    svr.Post("/predict", [&](const Request& req, Response& res) {
        if (!req.form.has_file("file")) {
            res.status = 400;
            res.set_content("No file uploaded", "text/plain");
            return;
        }

        const auto& file = req.form.get_file("file");
        string upload_path = "uploads/" + file.filename;
        
        // Ensure uploads directory exists
        system("mkdir -p uploads");

        ofstream ofs(upload_path, ios::binary);
        ofs << file.content;
        ofs.close();

        try {
            // Auto-train if not trained
            if (!g_state.trained) {
                runFullAnalysis("att_faces", 80, 0.8f, true, true);
            }
            json result = predictSingleImage(upload_path, true);
            res.set_content(result.dump(), "application/json");
        } catch (const exception& e) {
            res.status = 500;
            res.set_content(e.what(), "text/plain");
        }
    });

    // Endpoint for ROC plot
    svr.Get("/roc-plot", [&](const Request& req, Response& res) {
        ifstream ifs("roc_curve_cpp.png", ios::binary);
        if (ifs) {
            stringstream ss;
            ss << ifs.rdbuf();
            res.set_content(ss.str(), "image/png");
        } else {
            res.status = 404;
            res.set_content("ROC plot not found", "text/plain");
        }
    });

    cout << "Face Recognition System running on http://0.0.0.0:8000" << endl;
    
    /* Automatically open browser disabled to prevent terminal clutter
    #ifdef __linux__
    system("xdg-open http://localhost:8000 &");
    #endif
    */

    svr.listen("0.0.0.0", 8000);
}
