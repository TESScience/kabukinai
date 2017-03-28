SET(psf_unit_tests_SOURCES unit_tests.c psf.h)
ADD_EXECUTABLE(run_psf_unit_test ${psf_unit_tests_SOURCES})
TARGET_LINK_LIBRARIES(run_psf_unit_test psf)
SET_PROPERTY(TARGET run_psf_unit_test PROPERTY C_STANDARD 11)
ADD_TEST(psf_unit_test run_psf_unit_test)

SET(psf_benchmark_SOURCES benchmark.c psf.h)

ADD_EXECUTABLE(run_psf_benchmark ${psf_benchmark_SOURCES})
TARGET_LINK_LIBRARIES(run_psf_benchmark psf)
SET_PROPERTY(TARGET run_psf_benchmark PROPERTY C_STANDARD 11)
ADD_TEST(psf_benchmark run_psf_benchmark)
