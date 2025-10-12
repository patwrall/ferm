#include <cstdlib>
#include <exception>
#include <fmt/base.h>
#include <fmt/format.h>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <internal_use_only/config.hpp>
#include <string>

int main(int argc, const char **argv)
{
  try {
    CLI::App app{ fmt::format("{} version {}", ferm::cmake::project_name, ferm::cmake::project_version) };

    bool show_version = false;
    app.add_flag("--version", show_version, "Show version information");

    CLI11_PARSE(app, argc, argv);

    if (show_version) {
      fmt::print("{}\n", ferm::cmake::project_version);
      return EXIT_SUCCESS;
    }

  } catch (const std::exception &e) {
    spdlog::error("Unhandled exception in main: {}", e.what());
  }
}
