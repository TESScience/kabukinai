SET(TRY_SLICING_SOURCES try_slicing.cu httm.cu kabukinai.h)
SET(TRY_SMEAR_SOURCES try_smear.cu httm.cu kabukinai.h)
SET(TRY_NOISE_SOURCES try_noise.cu httm.cu kabukinai.h)

CONFIGURE_FILE(generate_star_test_data_tsv.py generate_star_test_data_tsv.py COPYONLY)

ADD_CUSTOM_COMMAND(
        OUTPUT star_test_data.tsv
        COMMAND python generate_star_test_data_tsv.py
        DEPENDS generate_star_test_data_tsv.py)

ADD_CUSTOM_COMMAND(
        OUTPUT psf.fits
        COMMAND psfgen psf.fits)

ADD_CUSTOM_TARGET(
        test_kabukinai
        COMMAND kabukinai psf.fits star_test_data.tsv simulated_image.fits
        DEPENDS star_test_data.tsv psf.fits kabukinai)

ADD_TEST(
        NAME kabukinai_smoke_test
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target test_kabukinai
)

CUDA_ADD_EXECUTABLE(try_slicing ${TRY_SLICING_SOURCES})
CUDA_ADD_EXECUTABLE(try_smear ${TRY_SMEAR_SOURCES})
CUDA_ADD_EXECUTABLE(try_noise ${TRY_NOISE_SOURCES})

ADD_TEST(test_slicing try_slicing)
ADD_TEST(test_smear try_smear)
ADD_TEST(test_noise try_noise)
