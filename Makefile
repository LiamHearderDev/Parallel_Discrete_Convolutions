# Name: Liam Hearder       Student Number: 23074422
# Name: Pranav Menon       Student Number: 24069351

CC = mpicc
CFLAGS = -fopenmp -Wall -Werror

SOURCE = conv2d.c
TARGET = conv2d

all:	$(TARGET)

$(TARGET):	$(SOURCE) 
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

rebuild:	clean all

.PHONY:	all clean rebuild