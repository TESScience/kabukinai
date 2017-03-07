#include "star_data.h"
#include <fstream>
#include <vector>

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
                          const minmax x_pixels,
                          const minmax y_pixels,
                          const long panels_per_side) {
    // TODO: error on negative
    // TODO: error if not evenly divisible by panels_per_side
    const long x_dimension = x_pixels.max - x_pixels.min;
    const long y_dimension = y_pixels.max - y_pixels.min;

    data->metadata.single_panel_pixel_dimensions.x_dimension = x_dimension / panels_per_side;
    data->metadata.single_panel_pixel_dimensions.y_dimension = y_dimension / panels_per_side;
    data->metadata.panel_indexes_dimensions.x_dimension = panels_per_side;
    data->metadata.panel_indexes_dimensions.y_dimension = panels_per_side;
    data->metadata.x_pixels = x_pixels;
    data->metadata.y_pixels = y_pixels;
    // TODO: error on negative
    const unsigned long number_of_panels = (const unsigned long) (panels_per_side * panels_per_side);
    std::vector<std::vector<star>> panel_intermediate_data(number_of_panels);
    for (const star &some_star : stars) {
        // TODO: Error if star out of bounds
        const unsigned long bin_x = (const unsigned long) (((some_star.x) - x_pixels.min) / panels_per_side);
        const unsigned long bin_y = (const unsigned long) (((some_star.y) - y_pixels.min) / panels_per_side);
        panel_intermediate_data.at(panel_index_lookup(bin_x, bin_y, data->metadata.panel_indexes_dimensions)).push_back(
                some_star);
    }
    data->panel_indexes = (long *) calloc(sizeof(long), number_of_panels + 1);
    data->stars = (star *) calloc(sizeof(star), stars.size());
    data->panel_indexes[number_of_panels] = stars.size();
    long panel_index = 0;
    for (unsigned long i = 0; i < number_of_panels; ++i) {
        data->panel_indexes[i] = panel_index;
        std::copy(panel_intermediate_data[i].begin(), panel_intermediate_data[i].end(), data->stars + panel_index);
        panel_index = panel_index + panel_intermediate_data[i].size();
    }

    return KABUKINAI_STAR_DATA_SUCCESS;
}

extern "C" {
int parse_star_data_from_tsv(star_data *data,
                             const char *file_name,
                             const minmax x_pixels,
                             const minmax y_pixels,
                             const long panels_per_side) {
    return star_data_from_vector(data, parse_star_vector_from_tsv(file_name), x_pixels, y_pixels, panels_per_side);
}
}
