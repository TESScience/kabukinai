#ifndef KABUKINAI_STAR_DATA_INTERNAL_HPP
#define KABUKINAI_STAR_DATA_INTERNAL_HPP

#include "star_data.h"
#include <vector>
#include <functional>

template<typename T>
int star_data_from_vector(star_data *data,
                          const std::vector<T> stars,
                          const dimensions image_dimensions,
                          const dimensions single_panel_pixel_dimensions,
                          const std::function<star_pixel_coordinate(T)> transform_input_to_star) {

    data->meta_data.single_panel_pixel_dimensions = single_panel_pixel_dimensions;
    data->meta_data.image_dimensions = image_dimensions;
    data->meta_data.panel_indices_dimensions.x_dimension =
            (int) ceil((float) image_dimensions.x_dimension / (float) single_panel_pixel_dimensions.x_dimension + 2);
    data->meta_data.panel_indices_dimensions.y_dimension =
            (int) ceil((float) image_dimensions.y_dimension / (float) single_panel_pixel_dimensions.y_dimension + 2);

    const star_meta_data &meta_data = data->meta_data;
    const unsigned long number_of_panels = (const unsigned long) (meta_data.panel_indices_dimensions.x_dimension *
                                                                  meta_data.panel_indices_dimensions.y_dimension);

    std::vector<std::vector<star_pixel_coordinate> > panel_intermediate_data(number_of_panels);
    for (const T &input : stars) {
        const star_pixel_coordinate input_star = transform_input_to_star(input);
        if (CHECK_PIXEL_VALID(input_star.point.x, input_star.point.y, meta_data)) {
            const int panel_index = PANEL_INDEX_LOOKUP_BY_PIXEL(input_star.point.x, input_star.point.y, meta_data);
            panel_intermediate_data.at((unsigned long) panel_index).push_back(input_star);
        }
    }
    data->panel_indices = (int *) calloc(sizeof(int), number_of_panels + 1);
    data->stars = (star_pixel_coordinate *) calloc(sizeof(star_pixel_coordinate), stars.size());
    data->panel_indices[number_of_panels] = (int) stars.size();
    int panel_index = 0;
    for (unsigned long i = 0; i < number_of_panels; ++i) {
        data->panel_indices[i] = panel_index;
        std::copy(panel_intermediate_data[i].begin(), panel_intermediate_data[i].end(), data->stars + panel_index);
        panel_index = panel_index + (int) panel_intermediate_data[i].size();
    }

    return KABUKINAI_STAR_DATA_SUCCESS;
}

#endif //KABUKINAI_STAR_DATA_INTERNAL_HPP
