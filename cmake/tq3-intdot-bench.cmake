function(add_tq3_intdot_bench)
    if (GGML_SYCL AND NOT TARGET test-tq3-4s-intdot-bench)
        add_executable(test-tq3-4s-intdot-bench "${PROJECT_SOURCE_DIR}/tests/test-tq3-4s-intdot-bench.cpp")
        target_include_directories(test-tq3-4s-intdot-bench PRIVATE "${PROJECT_SOURCE_DIR}/ggml/src")
        target_link_libraries(test-tq3-4s-intdot-bench PRIVATE ggml)
    endif()
endfunction()

cmake_language(DEFER CALL add_tq3_intdot_bench)
