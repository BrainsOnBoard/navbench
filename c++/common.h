#pragma once

// BoB robotics includes
#include "common/path.h"
#include "common/stopwatch.h"
#include "navigation/image_database.h"

// OpenCV
#include <opencv2/opencv.hpp>

// Standard C++ includes
#include <iostream>
#include <vector>

static const cv::Size ImageSize{ 90, 25 };
static const auto DatabaseRoot = BoBRobotics::Path::getProgramDirectory() /
                                 "../datasets/rc_car/Stanmer_park_dataset";
static const std::vector<std::string> TrainRoutes = { "0511/unwrapped_dataset1",
                                                      "0511/unwrapped_dataset2" };
const std::vector<std::string> TestRoutes = { "0511/unwrapped_dataset3" };

void
loadDatabaseImages(std::vector<cv::Mat> &images,
                   const std::string &dbName)
{
    const BoBRobotics::Navigation::ImageDatabase db{ DatabaseRoot / dbName };
    BOB_ASSERT(!db.empty());

    db.loadImages(images, ImageSize);
    std::cout << "Loaded " << images.size() << " images from " << dbName << "\n";
}

template<class Algo>
void
doTest(Algo &algo, std::vector<cv::Mat> &testImages, cv::FileStorage &fs)
{
    static std::vector<double> headings;
    headings.clear();
    BoBRobotics::Stopwatch timer;

    std::cout << "Testing...";
    timer.start();
    for (const auto &image : testImages) {
        const units::angle::degree_t heading = std::get<0>(algo.getHeading(image));
        headings.push_back(heading.value());
    }
    const units::time::millisecond_t testTime = timer.elapsed();
    std::cout << "Completed in " << testTime << "\n";

    fs << "time_per_image_ms" << testTime.value() / testImages.size()
       << "headings_deg"
       << "[" << headings << "]";
}

template<class Algo>
void
trainAndTest(Algo &algo)
{
    using namespace BoBRobotics;
    using namespace units::time;

    std::vector<cv::Mat> trainImages;
    for (const auto &route : TrainRoutes) {
        loadDatabaseImages(trainImages, route);
    }

    std::vector<cv::Mat> testImages;
    for (const auto &route : TestRoutes) {
        loadDatabaseImages(testImages, route);
    }

    std::string ofName = Path::getProgramPath().str() + ".yaml";
    cv::FileStorage fs{ ofName, cv::FileStorage::WRITE };
    fs << "data"
       << "{"
       << "bob_robotics_git_commit" << BOB_ROBOTICS_GIT_COMMIT
       << "bob_project_git_commit" << BOB_PROJECT_GIT_COMMIT
       << "image_size" << ImageSize;

    Stopwatch timer;
    std::cout << "Training...";
    timer.start();
    algo.trainRoute(trainImages);
    const millisecond_t trainTime = timer.elapsed();
    std::cout << "Completed in " << trainTime << "\n";

    fs << "training"
       << "{"
       << "routes"
       << "[" << TrainRoutes << "]"
       << "time_per_image_ms" << trainTime.value() / trainImages.size()
       << "}";

    fs << "testing"
       << "{"
       << "routes"
       << "[" << TestRoutes << "]";
    doTest(algo, testImages, fs);
    fs << "}";

    fs << "}";
}
