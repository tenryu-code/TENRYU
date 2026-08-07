#include "drivers/cli.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>

#include <unistd.h>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace tenryu::drivers {

namespace {
// Redirected-stdout sink: both spdlog stdout sinks (color and plain) end
// every log() with ::fflush -- correct on a terminal, but it defeats the
// enlarged stdout stdio buffer (main.cpp, TENRYU_LOG_FULLBUF_MB) whenever
// stdout is redirected to a file, recreating the network-FS per-line
// flush storm. This sink just fwrites into the stdio buffer; explicit
// flushes (flush_on(warn), the periodic flusher, shutdown) still drain it.
class buffered_stdout_sink final
    : public spdlog::sinks::base_sink<std::mutex> {
 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    spdlog::memory_buf_t formatted;
    formatter_->format(msg, formatted);
    std::fwrite(formatted.data(), 1, formatted.size(), stdout);
  }
  void flush_() override {
    std::fflush(stdout);
  }
};
}  // namespace

void add_common_cli_options(CLI::App& app, CliOptions& options) {
  app.add_flag("--verbose", options.verbose,
               "Enable verbose logging");
  app.add_flag("--quiet", options.quiet,
               "Reduce logging to warnings and errors");
}

void configure_logging(const CliOptions& options) {
  if (options.quiet) {
    spdlog::set_level(spdlog::level::warn);
  } else if (options.verbose) {
    spdlog::set_level(spdlog::level::debug);
  } else {
    spdlog::set_level(spdlog::level::info);
  }
}

void setup_file_logging(const std::string& log_dir) {
  const std::string log_path =
      (std::filesystem::path(log_dir) / "tenryu.log").string();
  // Terminal: keep the color sink (its per-line flush is what an
  // interactive user wants). Redirected: batch through the stdio buffer.
  spdlog::sink_ptr console_sink;
  if (::isatty(::fileno(stdout)) != 0) {
    console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  } else {
    console_sink = std::make_shared<buffered_stdout_sink>();
  }
  // W-R3 (file-sink leg of the W-R2 network-FS flush-storm fix): give the
  // log file's FILE* a large fully-buffered libc buffer so per-step
  // diagnostic lines do not drain to a (possibly network) filesystem every
  // few steps. TENRYU_LOG_FULLBUF_MB — shared with the stdout knob in
  // main.cpp — sizes the buffer (default 4.0 MiB, <=0 disables).
  // Freshness and crash context are preserved separately: flush_on(warn)
  // pushes warnings/errors/criticals through immediately (strictly better
  // than the previous no-flush-policy sink) and a periodic flusher
  // (TENRYU_LOG_FLUSH_EVERY_S, default 30, <=0 disables) bounds how stale
  // the log tail can get for external health checks.
  double fullbuf_mb = 4.0;
  if (const char* s = std::getenv("TENRYU_LOG_FULLBUF_MB")) {
    fullbuf_mb = std::atof(s);
  }
  spdlog::file_event_handlers handlers;
  if (fullbuf_mb > 0.0) {
    const std::size_t fullbuf_bytes =
        static_cast<std::size_t>(fullbuf_mb * 1024.0 * 1024.0);
    handlers.after_open = [fullbuf_bytes](const spdlog::filename_t&,
                                          std::FILE* file_stream) {
      // glibc ignores the size hint when buf is NULL (allocates
      // st_blksize on first I/O), so supply the buffer. Intentionally
      // leaked: the FILE* lives to process end and setup runs once.
      char* fullbuf = new char[fullbuf_bytes];
      (void)std::setvbuf(file_stream, fullbuf, _IOFBF, fullbuf_bytes);
    };
  }
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
      log_path, true, handlers);
  auto logger = std::make_shared<spdlog::logger>(
      "tenryu", spdlog::sinks_init_list{console_sink, file_sink});
  logger->set_level(spdlog::default_logger()->level());
  logger->flush_on(spdlog::level::warn);
  spdlog::set_default_logger(logger);
  double flush_every_s = 30.0;
  if (const char* s = std::getenv("TENRYU_LOG_FLUSH_EVERY_S")) {
    flush_every_s = std::atof(s);
  }
  if (flush_every_s > 0.0) {
    spdlog::flush_every(std::chrono::milliseconds(
        static_cast<long long>(flush_every_s * 1000.0)));
  }
  // Normal-exit completeness for the buffered stdout sink: drain the
  // enlarged stdout buffer while its storage (a function-local static in
  // main.cpp) is still alive — teardown ordering otherwise loses the tail
  // (observed: run.log truncated to pre-logger lines on a clean exit).
  // Registered after that static's construction, so it runs before the
  // static's destructor in the LIFO atexit order. Crash paths flush
  // explicitly in core/error.cpp.
  (void)std::atexit([] { std::fflush(stdout); });
}

}  // namespace tenryu::drivers
