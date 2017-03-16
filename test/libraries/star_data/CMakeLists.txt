SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic -g")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-long-long -pedantic -g")

configure_file(generate_star_test_data_tsv.py generate_star_test_data_tsv.py COPYONLY)

ADD_CUSTOM_COMMAND(
        OUTPUT star_test_data.tsv
        COMMAND python3 generate_star_test_data_tsv.py
        DEPENDS generate_star_test_data_tsv.py
)

SET(parse_star_data_tsv_SOURCES parse_star_data_tsv.c star_data.h)
ADD_EXECUTABLE(parse_star_data_tsv ${parse_star_data_tsv_SOURCES})
TARGET_LINK_LIBRARIES(parse_star_data_tsv star_data)


ADD_CUSTOM_TARGET(test_parse_star_data
        COMMAND parse_star_data_tsv star_test_data.tsv
        DEPENDS parse_star_data_tsv star_test_data.tsv)

ADD_TEST(
        NAME parse_star_data_unit_test
        COMMAND parse_star_data_tsv star_test_data.tsv
        DEPENDS parse_star_data_tsv star_test_data.tsv
)
