#include "star_data.h"
#include <fstream>
#include <vector>
#include <math.h>

class ParseStarDataException : public std::exception {
public:
    /** Constructor (C strings).
     *  @param message C-style string error message.
     *                 The string contents are copied upon construction.
     *                 Hence, responsibility for deleting the char* lies
     *                 with the caller.
     */
    explicit ParseStarDataException(const char *message) :
            msg_(message) {
    }

    /** Destructor.
     * Virtual to allow for subclassing.
     */
    virtual ~ParseStarDataException() throw() {}

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

std::vector<star> parse_star_vector_from_tsv(const char *file_name) {
    std::ifstream infile(file_name);
    std::vector<star> star_vector;
    std::string line;
    float intensities[8];
    while (std::getline(infile, line)) {
        const char *cline = line.c_str();
        star new_star;
        int parsed_arguments = sscanf(cline, "%g %g %g %g %g %g %g %g %g %g",
                                      &new_star.x,
                                      &new_star.y,
                                      &intensities[0], &intensities[1],
                                      &intensities[2], &intensities[3],
                                      &intensities[4], &intensities[5],
                                      &intensities[6], &intensities[7]);
        if (parsed_arguments != STAR_COLORS + 2) {
            char error_message_data[1024];
            snprintf(error_message_data, sizeof(error_message_data),
                     "Could not parse line of TSV:\n\n %s", cline);
            throw ParseStarDataException(error_message_data);
        }
        for (int color_index = 0; color_index < STAR_COLORS; ++color_index)
            new_star.intensities[color_index] = intensities[color_index];
        star_vector.push_back(new_star);
    }
    return star_vector;
}

int star_data_from_vector(star_data *data,
                          const std::vector<star> stars,
                          const dimensions image_dimensions,
                          const dimensions single_panel_pixel_dimensions) {

    // TODO: error on negative
    data->metadata.single_panel_pixel_dimensions = single_panel_pixel_dimensions;
    data->metadata.image_dimensions = image_dimensions;
    data->metadata.panel_indices_dimensions.x_dimension =
            (int) ceil(image_dimensions.x_dimension / single_panel_pixel_dimensions.x_dimension + 2);
    data->metadata.panel_indices_dimensions.y_dimension =
            (int) ceil(image_dimensions.y_dimension / single_panel_pixel_dimensions.y_dimension + 2);

    const unsigned long number_of_panels = (const unsigned long) (data->metadata.panel_indices_dimensions.x_dimension *
                                                                  data->metadata.panel_indices_dimensions.y_dimension);

    std::vector<std::vector<star>> panel_intermediate_data(number_of_panels);
    for (const star &star_data : stars) {
        // TODO: Error if star out of bounds
        panel_intermediate_data.at(
                (unsigned long) panel_index_lookup(star_data.x, star_data.y, data->metadata)).push_back(
                star_data);
    }
    data->panel_indices = (long *) calloc(sizeof(long), number_of_panels + 1);
    data->stars = (star *) calloc(sizeof(star), stars.size());
    data->panel_indices[number_of_panels] = stars.size();
    long panel_index = 0;
    for (unsigned long i = 0; i < number_of_panels; ++i) {
        data->panel_indices[i] = panel_index;
        std::copy(panel_intermediate_data[i].begin(), panel_intermediate_data[i].end(), data->stars + panel_index);
        panel_index = panel_index + panel_intermediate_data[i].size();
    }

    return KABUKINAI_STAR_DATA_SUCCESS;
}

extern "C" {
int parse_star_data_from_tsv(star_data *data,
                             const char *file_name,
                             const dimensions image_dimensions,
                             const dimensions single_panel_pixel_dimensions) {
    return star_data_from_vector(data, parse_star_vector_from_tsv(file_name), image_dimensions,
                                 single_panel_pixel_dimensions);
}
}
