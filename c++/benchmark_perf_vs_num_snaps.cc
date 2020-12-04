#include "common.h"

// BoB robotics includes
#include "navigation/perfect_memory.h"
#include "navigation/infomax.h"

using namespace BoBRobotics;

template<class Algo>
void
runTest(Algo &algo, std::vector<cv::Mat> &trainImages,
        std::vector<cv::Mat> &testImages, cv::FileStorage &fs)
{
    constexpr size_t imageStepMax = 50;
    for (size_t i = 1; i <= imageStepMax; i++) {
        std::cout << "Step: " << i << "\n";

        fs << "{"
           << "step" << (int) i;

        algo.clearMemory();
        for (size_t j = 0; j < trainImages.size(); j += i) {
            algo.train(trainImages[j]);
        }

        doTest(algo, testImages, fs);
        fs << "}";
    }
}

int
bobMain(int, char **)
{
    std::string ofName = Path::getProgramPath().str() + ".yaml";
    cv::FileStorage fs{ ofName, cv::FileStorage::WRITE };
    fs << "data"
       << "{"
       << "bob_robotics_git_commit" << BOB_ROBOTICS_GIT_COMMIT
       << "bob_project_git_commit" << BOB_PROJECT_GIT_COMMIT
       << "image_size" << ImageSize;

    std::vector<cv::Mat> trainImages, testImages;
    for (const auto &route : TrainRoutes) {
        loadDatabaseImages(trainImages, route);
    }
    fs << "training"
       << "{"
       << "routes" << TrainRoutes
       << "}";

    loadDatabaseImages(testImages, TestRoutes[0]);

    BOB_ASSERT(TestRoutes.size() == 1);
    fs << "testing"
       << "{"
       << "routes" << TestRoutes;

    {
        fs << "perfect_memory"
           << "[";
        Navigation::PerfectMemoryRotater<> pm{ ImageSize };
        runTest(pm, trainImages, testImages, fs);
        fs << "]";
    }

    {
        fs << "infomax"
           << "[";
        Navigation::InfoMaxRotater<> infomax{ ImageSize };
        runTest(infomax, trainImages, testImages, fs);
        fs << "]";
    }

    fs << "}"
       << "}";
    return EXIT_SUCCESS;
}
