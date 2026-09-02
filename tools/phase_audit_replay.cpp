#include <fstream>
#include <iostream>
#include <string>

#include "hydro/phase_audit.hpp"

int main(const int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: phase_audit_replay <ledger.jsonl>\n";
    return 2;
  }

  const std::string path = argv[1];
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::cerr << "phase_audit_replay: failed to open " << path << "\n";
    return 2;
  }

  const tenryu::hydro::phase_audit::ReplayResult result =
      tenryu::hydro::phase_audit::replay_jsonl(input);
  for (const std::string& violation : result.violations) {
    std::cerr << "FLAG: " << violation << "\n";
  }
  std::cout << "events=" << result.events_read
            << " violations=" << result.violations.size()
            << " truncated_final_line="
            << (result.truncated_final_line ? 1 : 0) << "\n";
  return result.passed() ? 0 : 1;
}
