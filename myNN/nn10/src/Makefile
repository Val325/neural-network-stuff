FILE = convneuralnet.cpp 
CFLAG = -Wall -g
CC = g++
INCLUDES = -I../include
LIBS = -Wl,-rpath,../lib -L../lib -lOpenImageIO 

conv:${OBJ}
	${CC} ${FILE} ${LIBS} ${CFLAGS} ${INCLUDES} -o $@ 

clean:
	-rm conv

.cpp.o:
	${CC} ${CFLAGS} ${INCLUDES} -c $<
