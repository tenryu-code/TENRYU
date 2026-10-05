#pragma once

#include <string>

namespace tenryu::drivers {

// The text `tenryu --version` prints after the program name: the version and
// the source revision the binary was built from, "1.0.0-beta.1 (source
// d23050a3b)". The revision comes from cmake/SourceRevision.cmake: the
// SOURCE_REVISION file of an exported tree, else the git commit of the source
// tree ("+modified" when tracked files differ from it); without either it is
// omitted. Only the tenryu executable compiles version_banner.cpp, so a new
// revision relinks the executable alone, not the libraries or the tests.
std::string version_banner();

}  // namespace tenryu::drivers
