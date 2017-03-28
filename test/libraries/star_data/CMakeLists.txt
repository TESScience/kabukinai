SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic -g")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic -g")

configure_file(generate_star_test_data_tsv.py generate_star_test_data_tsv.py COPYONLY)

ADD_CUSTOM_COMMAND(
        OUTPUT star_test_data.tsv
        COMMAND python generate_star_test_data_tsv.py
        DEPENDS generate_star_test_data_tsv.py
)

SET(parse_star_data_tsv_SOURCES parse_star_data_tsv.c star_data.h)
ADD_EXECUTABLE(parse_star_data_tsv ${parse_star_data_tsv_SOURCES})
SET_PROPERTY(TARGET parse_star_data_tsv PROPERTY C_STANDARD 11)
TARGET_LINK_LIBRARIES(parse_star_data_tsv star_data)

ADD_EXECUTABLE(run_test_coordinate_calculation test_coordinate_calculation.c star_data.h)
SET_PROPERTY(TARGET run_test_coordinate_calculation PROPERTY C_STANDARD 11)
TARGET_LINK_LIBRARIES(run_test_coordinate_calculation star_data)

ADD_TEST(
        NAME star_data_coordinate_calculation
        COMMAND run_test_coordinate_calculation
)

ADD_CUSTOM_TARGET(test_coordinate_calculation
        COMMAND run_test_coordinate_calculation
        DEPENDS run_test_coordinate_calculation)

ADD_CUSTOM_TARGET(test_parse_star_data
        COMMAND parse_star_data_tsv star_test_data.tsv
        DEPENDS parse_star_data_tsv star_test_data.tsv)

ADD_TEST(
        NAME parse_star_data_unit_test
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target test_parse_star_data
)
