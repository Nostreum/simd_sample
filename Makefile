CC=gcc-7 -O2 -msse3 -mavx -fopenmp
EXEC=hvadd
INCLUDE=-I include/
SOURCE_DIR=src/

all:
	$(CC) $(INCLUDE) -o $(EXEC) $(SOURCE_DIR)/matrixsum.c $(SOURCE_DIR)/main.c $(SOURCE_DIR)/hvadd.c $(SOURCE_DIR)/debug.c
