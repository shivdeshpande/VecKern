#pragma once
// Minimal test harness: no external dependency, so the tests build anywhere the library builds.
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace veckern::test {

struct Case {
    std::string name;
    std::function<void()> body;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

struct Register {
    Register(const char* name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};

inline int run_all() {
    int failed = 0;
    for (const auto& c : registry()) {
        try {
            c.body();
            std::printf("[ PASS ] %s\n", c.name.c_str());
        } catch (const std::exception& e) {
            ++failed;
            std::printf("[ FAIL ] %s: %s\n", c.name.c_str(), e.what());
        }
    }
    std::printf("%zu tests, %d failed\n", registry().size(), failed);
    return failed == 0 ? 0 : 1;
}

}  // namespace veckern::test

#define VK_CONCAT_(a, b) a##b
#define VK_CONCAT(a, b) VK_CONCAT_(a, b)
#define TEST_CASE(name)                                                                      \
    static void VK_CONCAT(test_fn_, __LINE__)();                                             \
    static veckern::test::Register VK_CONCAT(test_reg_, __LINE__)(name, VK_CONCAT(test_fn_, __LINE__)); \
    static void VK_CONCAT(test_fn_, __LINE__)()

#define REQUIRE(cond)                                                                                 \
    do {                                                                                              \
        if (!(cond)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                                              " REQUIRE(" #cond ")");                                 \
    } while (0)

#define REQUIRE_NEAR(a, b, tol)                                                                     \
    do {                                                                                            \
        const double va_ = (a), vb_ = (b);                                                          \
        if (std::fabs(va_ - vb_) > (tol))                                                           \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + \
                                     std::to_string(va_) + " vs " + std::to_string(vb_));           \
    } while (0)
