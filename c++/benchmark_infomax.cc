#include "common.h"

// BoB robotics includes
#define EXPOSE_INFOMAX_INTERNALS
#include "navigation/infomax.h"

int
bobMain(int, char **)
{
    using namespace BoBRobotics::Navigation;

    const auto weights = InfoMax<>::getInitialWeights(ImageSize.area(), ImageSize.area(), 42);
    InfoMaxRotater<> infomax{ ImageSize, weights };
    trainAndTest(infomax);

    return EXIT_SUCCESS;
}
