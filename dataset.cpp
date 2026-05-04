#include "dataset.h"
#include <dirent.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <map>

using namespace cv;
using namespace std;

void loadDataset(const string& path, Mat& images, Mat& labels) {
    CascadeClassifier face_cascade;
    string cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    if (!face_cascade.load(cascade_path)) {
        cerr << "Error: Could not load face cascade from " << cascade_path << endl;
        return;
    }

    DIR* dir = opendir(path.c_str());
    if (!dir) {
        cerr << "Error: Could not open directory " << path << endl;
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        string subfolder = entry->d_name;
        if (subfolder.empty() || subfolder[0] != 's') continue;

        try {
            int label = stoi(subfolder.substr(1));
            string subfolder_path = path + "/" + subfolder;

            DIR* subdir = opendir(subfolder_path.c_str());
            if (!subdir) continue;

            struct dirent* subentry;
            while ((subentry = readdir(subdir)) != NULL) {
                string filename = subentry->d_name;
                string ext = (filename.find_last_of(".") != string::npos) ? filename.substr(filename.find_last_of(".")) : "";
                for (auto &c : ext) c = tolower(c);
                
                if (ext != ".pgm" && ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") continue;

                string img_path = subfolder_path + "/" + filename;
                Mat img = imread(img_path, IMREAD_GRAYSCALE);
                if (img.empty()) continue;

                vector<Rect> faces;
                face_cascade.detectMultiScale(img, faces, 1.1, 4);

                Mat face_img;
                if (faces.size() > 0) {
                    face_img = img(faces[0]);
                } else {
                    face_img = img;
                }

                resize(face_img, face_img, Size(92, 112));
                
                Mat flattened;
                face_img.reshape(1, 1).convertTo(flattened, CV_32F);
                images.push_back(flattened);
                labels.push_back(label);
            }
            closedir(subdir);
        } catch (...) {
            continue;
        }
    }
    closedir(dir);
}

void splitDataset(const Mat& images, const Mat& labels, float train_ratio, Mat& train_images, Mat& train_labels, Mat& test_images, Mat& test_labels) {
    map<int, vector<int>> label_indices;
    for (int i = 0; i < labels.rows; ++i) {
        label_indices[labels.at<int>(i, 0)].push_back(i);
    }

    mt19937 g(42);
    for (auto const& [label, indices] : label_indices) {
        vector<int> current_indices = indices;
        shuffle(current_indices.begin(), current_indices.end(), g);

        int train_size = static_cast<int>(current_indices.size() * train_ratio);
        for (int i = 0; i < current_indices.size(); ++i) {
            if (i < train_size) {
                train_images.push_back(images.row(current_indices[i]));
                train_labels.push_back(label);
            } else {
                test_images.push_back(images.row(current_indices[i]));
                test_labels.push_back(label);
            }
        }
    }
}
