
todo: main
main:
	nvcc main.cu `pkg-config --libs --cflags opencv4` -o main

clean:
	rm main
