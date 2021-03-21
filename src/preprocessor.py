'''

This script normalizes the input file.
Reduce the dimension of the input from 32x32 to 8x8

Sample input:
00000000000001111000000000000000
00000000000011111110000000000000
00000000001111111111000000000000
00000001111111111111100000000000
00000001111111011111100000000000
00000011111110000011110000000000
00000011111110000000111000000000
00000011111110000000111100000000
00000011111110000000011100000000
00000011111110000000011100000000
00000011111100000000011110000000
00000011111100000000001110000000
00000011111100000000001110000000
00000001111110000000000111000000
00000001111110000000000111000000
00000001111110000000000111000000
00000001111110000000000111000000
00000011111110000000001111000000
00000011110110000000001111000000
00000011110000000000011110000000
00000001111000000000001111000000
00000001111000000000011111000000
00000001111000000000111110000000
00000001111000000001111100000000
00000000111000000111111000000000
00000000111100011111110000000000
00000000111111111111110000000000
00000000011111111111110000000000
00000000011111111111100000000000
00000000001111111110000000000000
00000000000111110000000000000000
00000000000011000000000000000000
 0

Sample output:
first 64 is the data while the last character is answer
0 1 6 15 12 1 0 0 0 7 16 6 6 10 0 0 0 8 16 2 0 11 2 0 0 5 16 3 0 5 7 0 0 7 13 3 0 8 7 0 0 4 12 0 1 13 5 0 0 0 14 9 15 9 0 0 0 0 6 14 7 1 0 0 0

'''

import os

#INPUT_FILE = "data" + os.sep + "raw-data.txt"
#OUTPUT_FILE = "data" + os.sep + "norm-data.txt"
INPUT_FILE = "data" + os.sep + "raw-sample.txt"
OUTPUT_FILE = "data" + os.sep + "norm-sample.txt"
RAW_WIDTH = 32
RAW_HEIGHT = 32
OUT_WIDTH = 8
OUT_HEIGHT = 8

input_file = open(INPUT_FILE, "r")
output_file = open(OUTPUT_FILE, "w")

row = 0
col = 0


tmp_input = []
result_line = []
result_arr = []
ans = []
chunk_size_w = int(RAW_WIDTH/OUT_WIDTH)
chunk_size_h = int(RAW_HEIGHT/OUT_HEIGHT)

for line in input_file:
    row += 1
    if row%(RAW_HEIGHT + 1) == 0:
        ans.append(int(line.strip()))     # reading the ans

        for i in range(OUT_HEIGHT):
            for j in range(OUT_WIDTH):
                chunk = 0
                for x in range(chunk_size_w):
                    for y in range(chunk_size_h):
                        chunk += int(tmp_input[y+i*chunk_size_h][x+j*chunk_size_w])

                result_line.append(chunk)
        result_arr.append(result_line)
        tmp_input = []
        result_line = []
    else:
        tmp_input.append(line.strip())

output_count = OUT_WIDTH * OUT_HEIGHT
output_file.write(str(len(result_arr)) + ' ' + str(output_count) + '\n')
for i in range(len(result_arr)):
    line = result_arr[i]
    for num in line:
        output_file.write(str(num) + " ")
    output_file.write(str(ans[i]) + "\n")

input_file.close()
output_file.close()



