#!/usr/bin/env python3

offset = -16
panel_height = 67
panel_width = 67
stride = 32

with open('star_test_data.tsv', 'w') as output:
    for line in ("\t".join(map(str, (i,j,1)))
                 for i in range(offset, panel_width*stride, stride)
                 for j in range(offset, panel_height*stride, stride)):
        output.write("{}\n".format(line))