SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic -g")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic -g")

configure_file(generate_star_test_data_tsv.py generate_star_test_data_tsv.py COPYONLY)

ADD_CUSTOM_COMMAND(
        OUTPUT star_test_data.tsv
        COMMAND python3 generate_star_test_data_tsv.py
        DEPENDS generate_star_test_data_tsv.py)

ADD_CUSTOM_COMMAND(
        OUTPUT psf.fits
        COMMAND psfgen psf.fits)

ADD_CUSTOM_TARGET(
        test_kabukinai
        COMMAND kabukinai psf.fits star_test_data.tsv simulated_image.fits
        DEPENDS star_test_data.tsv psf.fits kabukinai)

