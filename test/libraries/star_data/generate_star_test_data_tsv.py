#!/usr/bin/env python

offset = -16
panel_width = 66
panel_height = 67
stride = 32

with open('star_test_data.tsv', 'w') as output:
    for line in ("\t".join(map(str, (i,j,1)))
                 for i in range(offset, panel_width*stride + offset, stride)
                 for j in range(offset, panel_height*stride + offset, stride)):
        output.write("{}\n".format(line))
