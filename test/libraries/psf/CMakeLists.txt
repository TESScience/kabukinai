ADD_EXECUTABLE(psf_unit_tests unit_tests.c psf.h)
TARGET_LINK_LIBRARIES(psf_unit_tests libpsf)
ADD_TEST(psf_test psf_unit_tests)
