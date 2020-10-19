
todo: main
main:
	nvcc main.cu `pkg-config --libs --cflags opencv` -o main

clean:
	rm main
