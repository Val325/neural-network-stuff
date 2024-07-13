FILE = convneuralnet.cpp 
CFLAG = -ggdb3 -g
CC = g++
INCLUDES = -I../include
LIBS = -Wl,-rpath,../lib -L../lib -lOpenImageIO 

conv:${OBJ}
	${CC} ${FILE} ${LIBS} ${CFLAG} ${INCLUDES} -o $@ 

clean:
	-rm conv

.cpp.o:
	${CC} ${CFLAGS} ${INCLUDES} -c $<
