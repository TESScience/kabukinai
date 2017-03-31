#include "star_data.h"
#include <vector>
#include <functional>
#include <fstream>

class StarDataException : public std::exception {
public:
    /** Constructor (C strings).
     *  @param message C-style string error message.
     *                 The string contents are copied upon construction.
     *                 Hence, responsibility for deleting the char* lies
     *                 with the caller.
     */
    explicit StarDataException(const char *message) :
            msg_(message) {
    }

    /** Destructor.
     * Virtual to allow for subclassing.
     */
    virtual ~StarDataException() throw() {}

    /** Returns a pointer to the (constant) error description.
     *  @return A pointer to a const char*. The underlying memory
     *          is in posession of the Exception object. Callers must
     *          not attempt to free the memory.
     */
    virtual const char *what() const throw() {
        return msg_.c_str();
    }

protected:
    /** Error message.
     */
    std::string msg_;
};

std::vector<star_pixel_coordinate> parse_star_vector_from_tsv(const char *file_name) {
    std::ifstream infile(file_name);
    std::vector<star_pixel_coordinate> star_vector;
    std::string line;
    float intensities[8];
    while (std::getline(infile, line)) {
        const char *cline = line.c_str();
        star_pixel_coordinate new_star;
        int parsed_arguments = sscanf(cline, "%g %g %g %g %g %g %g %g %g %g",
                                      &new_star.point.x,
                                      &new_star.point.y,
                                      &intensities[0], &intensities[1],
                                      &intensities[2], &intensities[3],
                                      &intensities[4], &intensities[5],
                                      &intensities[6], &intensities[7]);
        if (parsed_arguments != STAR_COLORS + 2) {
            char error_message_data[1024];
            snprintf(error_message_data, sizeof(error_message_data),
                     "Could not parse line of TSV:\n\n %s", cline);
            throw StarDataException(error_message_data);
        }
        for (int color_index = 0; color_index < STAR_COLORS; ++color_index)
            new_star.intensities[color_index] = intensities[color_index];
        star_vector.push_back(new_star);
    }
    return star_vector;
}


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

extern "C" {
int parse_star_data_from_tsv(star_data *data,
                             const char *file_name,
                             const dimensions image_dimensions,
                             const dimensions single_panel_pixel_dimensions) {
    return star_data_from_vector<star_pixel_coordinate>(data,
                                       parse_star_vector_from_tsv(file_name),
                                       image_dimensions,
                                       single_panel_pixel_dimensions,
                                       [](star_pixel_coordinate x) -> star_pixel_coordinate { return x; });
}
}
