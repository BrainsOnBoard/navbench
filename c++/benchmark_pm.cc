#include "common.h"

// BoB robotics includes
#include "navigation/perfect_memory.h"

int
bobMain(int, char **)
{
    using namespace BoBRobotics::Navigation;

    PerfectMemoryRotater<> pm{ ImageSize };
    doTest(pm);

    return EXIT_SUCCESS;
}
