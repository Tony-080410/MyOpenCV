#pragma once
#include <stdexcept>
#include <string>

#define MYCV_ASSERT(cond)                                                                          \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            throw std::runtime_error(std::string("MyOpenCV: ") + #cond +                           \
                                     " failed at " __FILE__ ":" + std::to_string(__LINE__));       \
        }                                                                                          \
    } while (0)
