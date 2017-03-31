#include "star_data_internal.hpp"
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

std::vector<star_pixel_coordinate> parse_star_pixel_coordinate_vector_from_tsv(const char *file_name) {
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



extern "C" {
int parse_star_data_from_tsv(star_data *data,
                             const char *file_name,
                             const dimensions image_dimensions,
                             const dimensions single_panel_pixel_dimensions) {
    return star_data_from_vector<star_pixel_coordinate>(
            data,
            parse_star_pixel_coordinate_vector_from_tsv(file_name),
            image_dimensions,
            single_panel_pixel_dimensions,
            [](star_pixel_coordinate x) -> star_pixel_coordinate { return x; });
}
}
